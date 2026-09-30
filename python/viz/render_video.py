#!/usr/bin/env python3
"""
Render a Chess-RL self-play game as a polished MP4 video.

The network plays a game against itself while every position, policy
distribution, and value estimate is recorded. Each ply is then rendered
as an animated sequence: pieces slide between squares, the top policy
candidates are drawn as glowing arrows scaled by probability, and a side
panel shows the candidate distribution, the value-head evaluation gauge,
and the move history.

Usage:
    python python/viz/render_video.py [--out selfplay.mp4] [--plies 60]
                                      [--seed 7] [--fps 24]

Requires pillow and an ffmpeg binary (the imageio-ffmpeg wheel provides
one if ffmpeg is not on PATH).
"""
import argparse
import os
import random
import shutil
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'train')))

import torch
import torch.nn.functional as F

from model.model import ChessNet
from python_chess import (SimpleChessBoard, encode_simple_board,
                          PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING)
from selfplay import _uci_to_index

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
ASSETS = os.path.join(ROOT, 'assets')

# ---------------------------------------------------------------- layout ----
W, H = 1280, 720
SQ = 74                       # square size
BOARD_X, BOARD_Y = 56, 84     # top-left of the board
BOARD_S = SQ * 8
GAUGE_X = BOARD_X + BOARD_S + 16
PANEL_X = GAUGE_X + 44
PANEL_W = W - PANEL_X - 40

# ---------------------------------------------------------------- palette ---
BG_TOP = (16, 19, 28)
BG_BOT = (30, 34, 48)
LIGHT_SQ = (222, 208, 182)
DARK_SQ = (140, 112, 92)
BOARD_EDGE = (10, 12, 18)
ACCENT = (108, 190, 255)
ACCENT_WARM = (255, 176, 92)
TEXT = (232, 236, 244)
TEXT_DIM = (148, 156, 172)
LAST_MOVE = (246, 220, 96)
CHECK_RED = (255, 84, 72)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


PIECE_LETTER = {PAWN: 'P', KNIGHT: 'N', BISHOP: 'B', ROOK: 'R', QUEEN: 'Q', KING: 'K'}


def load_piece_images(size):
    images = {}
    for value, letter in PIECE_LETTER.items():
        for sign, color in ((1, 'w'), (-1, 'b')):
            img = Image.open(os.path.join(ASSETS, f"{color}{letter}.png")).convert('RGBA')
            images[sign * value] = img.resize((size, size), Image.LANCZOS)
    return images


def ease(t):
    """Smooth ease-in-out."""
    return t * t * (3.0 - 2.0 * t)


def sq_center(rank, file):
    """Pixel center of a square; rank 7 is drawn at the top."""
    x = BOARD_X + file * SQ + SQ // 2
    y = BOARD_Y + (7 - rank) * SQ + SQ // 2
    return x, y


def uci_squares(uci):
    ff, fr = ord(uci[0]) - ord('a'), int(uci[1]) - 1
    tf, tr = ord(uci[2]) - ord('a'), int(uci[3]) - 1
    return (fr, ff), (tr, tf)


# ---------------------------------------------------------------- game ------
def play_game(net, max_plies, temperature, seed):
    """Self-play a game, recording everything the renderer needs."""
    rng = random.Random(seed)
    torch.manual_seed(seed)
    board = SimpleChessBoard()
    plies = []
    result = None

    for _ in range(max_plies):
        legal = board.get_legal_moves()
        if not legal:
            break
        state = encode_simple_board(board)
        with torch.no_grad():
            logits, value = net(state)

        mask = torch.full_like(logits, float('-inf'))
        for uci in legal:
            idx = _uci_to_index(uci)
            mask[0, idx] = logits[0, idx]
        probs = F.softmax(mask / temperature, dim=1)[0]

        # Aggregate probabilities per legal move (promotions share an index)
        move_probs = {}
        for uci in legal:
            move_probs[uci] = float(probs[_uci_to_index(uci)])
        total = sum(move_probs.values()) or 1.0
        move_probs = {k: v / total for k, v in move_probs.items()}

        chosen = rng.choices(list(move_probs), weights=list(move_probs.values()))[0]
        top = sorted(move_probs.items(), key=lambda kv: -kv[1])[:5]

        captured_sq = None
        (fr, ff), (tr, tf) = uci_squares(chosen)
        if board.board[tr, tf] != 0:
            captured_sq = (tr, tf)
        elif abs(board.board[fr, ff]) == PAWN and ff != tf:
            captured_sq = (fr, tf)  # en passant

        plies.append({
            'board': board.board.copy(),
            'white_to_move': board.white_to_move,
            'move': chosen,
            'top': top,
            'value': float(value.item()),
            'captured_sq': captured_sq,
        })
        board.apply_move(chosen)
        done, res = board.is_game_over()
        if done:
            result = res
            break

    final = {'board': board.board.copy(), 'white_to_move': board.white_to_move}
    return plies, final, result or "…"


# ---------------------------------------------------------------- drawing ---
def make_background():
    """Vertical gradient with a soft vignette, drawn once."""
    grad = np.linspace(0, 1, H)[:, None]
    top = np.array(BG_TOP, dtype=float)
    bot = np.array(BG_BOT, dtype=float)
    rows = top * (1 - grad) + bot * grad
    arr = np.repeat(rows[:, None, :], W, axis=1).astype(np.uint8)
    bg = Image.fromarray(arr, 'RGB')

    glow = Image.new('L', (W, H), 0)
    d = ImageDraw.Draw(glow)
    d.ellipse([W * 0.05, -H * 0.4, W * 0.75, H * 0.55], fill=46)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    tint = Image.new('RGB', (W, H), (64, 96, 150))
    bg = Image.composite(Image.blend(bg, tint, 0.35), bg, glow)
    return bg


def draw_board_base(draw):
    draw.rounded_rectangle(
        [BOARD_X - 10, BOARD_Y - 10, BOARD_X + BOARD_S + 10, BOARD_Y + BOARD_S + 10],
        radius=14, fill=BOARD_EDGE)
    for rank in range(8):
        for file in range(8):
            x = BOARD_X + file * SQ
            y = BOARD_Y + (7 - rank) * SQ
            color = LIGHT_SQ if (rank + file) % 2 else DARK_SQ
            draw.rectangle([x, y, x + SQ, y + SQ], fill=color)
    coord_font = font(13, bold=True)
    for file in range(8):
        draw.text((BOARD_X + file * SQ + SQ - 12, BOARD_Y + BOARD_S - 17),
                  chr(ord('a') + file), font=coord_font,
                  fill=DARK_SQ if file % 2 else LIGHT_SQ)
    for rank in range(8):
        draw.text((BOARD_X + 4, BOARD_Y + (7 - rank) * SQ + 3),
                  str(rank + 1), font=coord_font,
                  fill=LIGHT_SQ if rank % 2 else DARK_SQ)


def highlight_square(overlay_draw, rank, file, color, alpha):
    x = BOARD_X + file * SQ
    y = BOARD_Y + (7 - rank) * SQ
    overlay_draw.rectangle([x, y, x + SQ, y + SQ], fill=color + (alpha,))


def draw_arrow(overlay, from_sq, to_sq, color, prob, emphasize=False):
    """Glowing probability arrow between two squares."""
    x1, y1 = sq_center(*from_sq)
    x2, y2 = sq_center(*to_sq)
    vec = np.array([x2 - x1, y2 - y1], dtype=float)
    length = np.hypot(*vec)
    if length < 1:
        return
    u = vec / length
    x1, y1 = np.array([x1, y1]) + u * SQ * 0.28
    tip = np.array([x2, y2]) - u * SQ * 0.16
    head = SQ * (0.30 + 0.16 * prob)
    base = tip - u * head
    perp = np.array([-u[1], u[0]])

    width = max(5, int(SQ * (0.12 + 0.20 * prob)))
    alpha = int(190 + 60 * prob) if emphasize else int(90 + 110 * prob)

    layer = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.line([tuple((x1, y1)), tuple(base)], fill=color + (alpha,), width=width)
    wing = perp * (width * 1.35)
    d.polygon([tuple(tip), tuple(base + wing), tuple(base - wing)],
              fill=color + (alpha,))
    if emphasize:
        glow = layer.filter(ImageFilter.GaussianBlur(6))
        overlay.alpha_composite(glow)
    overlay.alpha_composite(layer)


def find_king(board_arr, white):
    target = KING if white else -KING
    pos = np.argwhere(board_arr == target)
    return tuple(pos[0]) if len(pos) else None


def in_check(board_arr, white):
    tmp = SimpleChessBoard()
    tmp.board = board_arr.copy()
    king = find_king(board_arr, white)
    if king is None:
        return False
    from python_chess import coords_to_sq
    return tmp._is_attacked_on_board(board_arr, coords_to_sq(*king), not white)


def draw_panel(draw, ply_data, ply_index, total_plies, history, shown_value):
    y = BOARD_Y - 6
    draw.text((PANEL_X, y), "POLICY NETWORK", font=font(15, bold=True), fill=ACCENT)
    mover = "White" if ply_data['white_to_move'] else "Black"
    draw.text((PANEL_X + PANEL_W - 132, y), f"ply {ply_index + 1}/{total_plies}",
              font=font(14), fill=TEXT_DIM)
    y += 30
    draw.text((PANEL_X, y), f"{mover} to move · sampling T=1.2",
              font=font(13), fill=TEXT_DIM)
    y += 30

    # Candidate probability bars
    bar_w = PANEL_W - 120
    p_max = max(p for _, p in ply_data['top']) or 1.0
    for uci, p in ply_data['top']:
        chosen = uci == ply_data['move']
        label_font = font(15, bold=chosen)
        draw.text((PANEL_X, y), uci, font=label_font,
                  fill=TEXT if chosen else TEXT_DIM)
        bx = PANEL_X + 66
        draw.rounded_rectangle([bx, y + 3, bx + bar_w, y + 15], radius=6,
                               fill=(255, 255, 255, 18) if not chosen else (255, 255, 255, 26))
        fill_w = max(8, int(bar_w * 0.85 * p / p_max))
        color = ACCENT_WARM if chosen else (86, 110, 150)
        draw.rounded_rectangle([bx, y + 3, bx + fill_w, y + 15], radius=6, fill=color)
        draw.text((bx + bar_w + 10, y), f"{p * 100:4.1f}%", font=font(13),
                  fill=TEXT if chosen else TEXT_DIM)
        y += 26

    # Value head
    y += 16
    draw.text((PANEL_X, y), "VALUE HEAD", font=font(15, bold=True), fill=ACCENT)
    y += 26
    vx, vw = PANEL_X, PANEL_W - 70
    draw.rounded_rectangle([vx, y + 2, vx + vw, y + 16], radius=7, fill=(255, 255, 255, 16))
    mid = vx + vw // 2
    draw.line([mid, y, mid, y + 18], fill=TEXT_DIM, width=1)
    v = max(-1.0, min(1.0, shown_value))
    end = mid + int((vw // 2 - 4) * v)
    color = (120, 214, 148) if v >= 0 else (240, 120, 110)
    draw.rounded_rectangle([min(mid, end), y + 4, max(mid, end), y + 14],
                           radius=5, fill=color)
    draw.text((vx + vw + 12, y - 1), f"{shown_value:+.2f}", font=font(15, bold=True),
              fill=color)
    y += 40

    # Move history
    draw.text((PANEL_X, y), "GAME", font=font(15, bold=True), fill=ACCENT)
    y += 26
    recent = history[-16:]
    start_no = len(history) - len(recent)
    col, row = 0, 0
    for i, mv in enumerate(recent):
        n = start_no + i
        label = f"{n // 2 + 1}.{'' if n % 2 == 0 else '..'}{mv}"
        tx = PANEL_X + col * ((PANEL_W // 2) + 6)
        draw.text((tx, y + row * 22), label,
                  font=font(13, bold=(i == len(recent) - 1)),
                  fill=TEXT if i == len(recent) - 1 else TEXT_DIM)
        col += 1
        if col == 2:
            col, row = 0, row + 1


def draw_eval_gauge(img, draw, shown_value):
    """Vertical white/black advantage gauge beside the board."""
    gy, gh, gw = BOARD_Y, BOARD_S, 20
    draw.rounded_rectangle([GAUGE_X, gy, GAUGE_X + gw, gy + gh], radius=9,
                           fill=(8, 9, 13))
    v = max(-1.0, min(1.0, shown_value))
    white_frac = 0.5 + v / 2
    split = gy + gh - int(gh * white_frac)
    draw.rounded_rectangle([GAUGE_X + 2, split, GAUGE_X + gw - 2, gy + gh - 2],
                           radius=7, fill=(236, 238, 242))
    draw.line([GAUGE_X + 2, gy + gh // 2, GAUGE_X + gw - 2, gy + gh // 2],
              fill=(120, 126, 140), width=1)


def render_frame(bg, pieces, board_arr, *, last_move=None, moving=None,
                 fade_sq=None, fade_alpha=255, arrows=None, check_sq=None,
                 panel=None, header_alpha=255):
    img = bg.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    # Header
    title_font = font(26, bold=True)
    draw.text((BOARD_X - 4, 26), "CHESS-RL", font=title_font, fill=TEXT)
    tw = draw.textlength("CHESS-RL", font=title_font)
    draw.text((BOARD_X + tw + 10, 33), "· neural self-play", font=font(17),
              fill=TEXT_DIM)
    tag = "policy + value network · AlphaZero-style training loop"
    draw.text((W - 40 - draw.textlength(tag, font=font(14)), 34), tag,
              font=font(14), fill=TEXT_DIM)

    draw_board_base(draw)

    overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    if last_move:
        for r, f in last_move:
            highlight_square(odraw, r, f, LAST_MOVE, 70)
    if check_sq:
        highlight_square(odraw, check_sq[0], check_sq[1], CHECK_RED, 90)

    if arrows:
        for (fr_sq, to_sq, p, emph) in arrows:
            color = ACCENT_WARM if emph else ACCENT
            draw_arrow(overlay, fr_sq, to_sq, color, p, emphasize=emph)
    img.alpha_composite(overlay) if img.mode == 'RGBA' else img.paste(
        overlay, (0, 0), overlay)

    # Pieces
    psize = pieces[PAWN].width
    off = (SQ - psize) // 2
    skip = set()
    if moving:
        skip.add(moving['from'])
    for rank in range(8):
        for file in range(8):
            piece = board_arr[rank, file]
            if piece == 0 or (rank, file) in skip:
                continue
            alpha_img = pieces[piece]
            if fade_sq == (rank, file) and fade_alpha < 255:
                alpha_img = alpha_img.copy()
                alpha_img.putalpha(alpha_img.getchannel('A').point(
                    lambda a: a * fade_alpha // 255))
            x = BOARD_X + file * SQ + off
            y = BOARD_Y + (7 - rank) * SQ + off
            img.paste(alpha_img, (x, y), alpha_img)

    if moving:
        (fr, ff), (tr, tf) = moving['from'], moving['to']
        t = ease(moving['t'])
        x1, y1 = BOARD_X + ff * SQ + off, BOARD_Y + (7 - fr) * SQ + off
        x2, y2 = BOARD_X + tf * SQ + off, BOARD_Y + (7 - tr) * SQ + off
        x = int(x1 + (x2 - x1) * t)
        y = int(y1 + (y2 - y1) * t)
        piece_img = pieces[moving['piece']]
        shadow = Image.new('RGBA', piece_img.size, (0, 0, 0, 0))
        shadow.paste((0, 0, 0, 90), (0, 0), piece_img)
        img.paste(shadow, (x + 3, y + 5), shadow)
        img.paste(piece_img, (x, y), piece_img)

    if panel:
        draw_eval_gauge(img, draw, panel['shown_value'])
        draw_panel(draw, panel['ply'], panel['index'], panel['total'],
                   panel['history'], panel['shown_value'])

    if header_alpha < 255:
        veil = Image.new('RGBA', img.size, (BG_TOP[0], BG_TOP[1], BG_TOP[2],
                                            255 - header_alpha))
        img.paste(veil, (0, 0), veil)
    return img


def title_card(bg, subtitle, alpha):
    img = bg.copy()
    draw = ImageDraw.Draw(img, 'RGBA')
    big = font(64, bold=True)
    small = font(22)
    tw = draw.textlength("CHESS-RL", font=big)
    draw.text(((W - tw) / 2, H / 2 - 90), "CHESS-RL", font=big,
              fill=TEXT + (alpha,) if False else TEXT)
    sw = draw.textlength(subtitle, font=small)
    draw.text(((W - sw) / 2, H / 2), subtitle, font=small, fill=TEXT_DIM)
    line_w = int(tw * 0.9)
    draw.rounded_rectangle([(W - line_w) / 2, H / 2 - 108,
                            (W + line_w) / 2, H / 2 - 104], radius=2, fill=ACCENT)
    if alpha < 255:
        veil = Image.new('RGBA', img.size, BG_TOP + (255 - alpha,))
        img.paste(veil, (0, 0), veil)
    return img


# ---------------------------------------------------------------- encode ----
def ffmpeg_binary():
    exe = shutil.which('ffmpeg')
    if exe:
        return exe
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='selfplay.mp4')
    parser.add_argument('--plies', type=int, default=60)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--fps', type=int, default=24)
    parser.add_argument('--model', default=None,
                        help='optional model_*.pt checkpoint to load')
    args = parser.parse_args()

    net = ChessNet()
    if args.model and os.path.exists(args.model):
        net.load_state_dict(torch.load(args.model, map_location='cpu',
                                       weights_only=True))
        print(f"Loaded model weights from {args.model}")
    net.eval()

    print("Playing self-play game…")
    plies, final, result = play_game(net, args.plies, 1.2, args.seed)
    print(f"Recorded {len(plies)} plies, result: {result}")

    bg = make_background()
    pieces = load_piece_images(int(SQ * 0.9))

    proc = subprocess.Popen(
        [ffmpeg_binary(), '-y', '-loglevel', 'error',
         '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}',
         '-r', str(args.fps), '-i', '-',
         '-c:v', 'libx264', '-crf', '20', '-preset', 'medium',
         '-pix_fmt', 'yuv420p', '-movflags', '+faststart', args.out],
        stdin=subprocess.PIPE)

    def emit(img, times=1):
        data = img.convert('RGB').tobytes()
        for _ in range(times):
            proc.stdin.write(data)

    # Intro
    for i in range(30):
        emit(title_card(bg, "a neural network learning chess through self-play",
                        int(255 * ease(min(1.0, i / 18)))))
    emit(title_card(bg, "a neural network learning chess through self-play", 255), 20)

    history = []
    shown_value = 0.0
    SLIDE, THINK, HOLD = 9, 8, 5

    for idx, ply in enumerate(plies):
        board_arr = ply['board']
        (fr_sq, to_sq) = uci_squares(ply['move'])
        piece = board_arr[fr_sq]
        last = uci_squares(history_move) if (history_move := history and history[-1]) else None
        check_sq = find_king(board_arr, ply['white_to_move']) \
            if in_check(board_arr, ply['white_to_move']) else None

        p_max = max(p for _, p in ply['top']) or 1.0
        arrows = [(uci_squares(u)[0], uci_squares(u)[1], p / p_max,
                   u == ply['move'])
                  for u, p in reversed(ply['top'])]

        # Thinking: show candidate arrows, ease the eval gauge
        for i in range(THINK):
            shown_value += (ply['value'] - shown_value) * 0.35
            panel = {'ply': ply, 'index': idx, 'total': len(plies),
                     'history': history, 'shown_value': shown_value}
            emit(render_frame(bg, pieces, board_arr, last_move=last,
                              arrows=arrows, check_sq=check_sq, panel=panel))

        # Move animation
        for i in range(SLIDE):
            t = (i + 1) / SLIDE
            fade = int(255 * (1 - t)) if ply['captured_sq'] else 255
            panel = {'ply': ply, 'index': idx, 'total': len(plies),
                     'history': history, 'shown_value': shown_value}
            emit(render_frame(
                bg, pieces, board_arr,
                last_move=[fr_sq, to_sq],
                moving={'from': fr_sq, 'to': to_sq, 'piece': piece, 't': t},
                fade_sq=ply['captured_sq'], fade_alpha=fade,
                check_sq=check_sq, panel=panel))

        history.append(ply['move'])

        # Settle on the resulting position
        next_board = plies[idx + 1]['board'] if idx + 1 < len(plies) else final['board']
        next_white = plies[idx + 1]['white_to_move'] if idx + 1 < len(plies) \
            else final['white_to_move']
        check2 = find_king(next_board, next_white) if in_check(next_board, next_white) else None
        panel = {'ply': ply, 'index': idx, 'total': len(plies),
                 'history': history, 'shown_value': shown_value}
        emit(render_frame(bg, pieces, next_board, last_move=[fr_sq, to_sq],
                          check_sq=check2, panel=panel), HOLD)

    # Outro
    outcome = {"1-0": "White wins", "0-1": "Black wins",
               "1/2-1/2": "Draw"}.get(result, f"Game paused after {len(plies)} plies")
    for i in range(30):
        emit(title_card(bg, f"{outcome} · every move chosen by the policy network",
                        int(255 * ease(min(1.0, i / 18)))))
    emit(title_card(bg, f"{outcome} · every move chosen by the policy network", 255), 36)

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    size = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} ({size:.1f} MB)")


if __name__ == '__main__':
    main()
