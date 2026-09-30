#!/usr/bin/env python3
"""
Render a Chess-RL self-play game as an MP4 video in a clean,
white-background, research-figure style.

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
from PIL import Image, ImageDraw, ImageFont

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
# White-background, print-style figure aesthetic. Candidate/chosen colors are
# the Okabe-Ito blue and vermillion (colorblind-safe on a white surface).
SURFACE = (252, 252, 251)
LIGHT_SQ = (244, 242, 237)
DARK_SQ = (181, 177, 166)
BOARD_EDGE = (60, 60, 60)
ACCENT = (0, 114, 178)        # candidate moves (Okabe-Ito blue)
ACCENT_WARM = (213, 94, 0)    # chosen move (Okabe-Ito vermillion)
TEXT = (26, 26, 26)
TEXT_DIM = (110, 110, 110)
HAIRLINE = (208, 206, 202)
BAR_TRACK = (236, 235, 232)
LAST_MOVE = (240, 210, 80)
CHECK_RED = (197, 58, 50)

FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def font(size, bold=False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def serif(size, bold=False):
    name = "DejaVuSerif-Bold.ttf" if bold else "DejaVuSerif.ttf"
    return ImageFont.truetype(os.path.join(FONT_DIR, name), size)


def mono(size, bold=False):
    name = "DejaVuSansMono-Bold.ttf" if bold else "DejaVuSansMono.ttf"
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
    """Plain paper-white surface, drawn once."""
    return Image.new('RGB', (W, H), SURFACE)


def draw_board_base(draw):
    for rank in range(8):
        for file in range(8):
            x = BOARD_X + file * SQ
            y = BOARD_Y + (7 - rank) * SQ
            color = LIGHT_SQ if (rank + file) % 2 else DARK_SQ
            draw.rectangle([x, y, x + SQ, y + SQ], fill=color)
    # Hairline frame, as in a printed diagram
    draw.rectangle([BOARD_X - 1, BOARD_Y - 1,
                    BOARD_X + BOARD_S, BOARD_Y + BOARD_S],
                   outline=BOARD_EDGE, width=1)
    # File and rank labels outside the board
    coord_font = font(12)
    for file in range(8):
        c = chr(ord('a') + file)
        cw = draw.textlength(c, font=coord_font)
        draw.text((BOARD_X + file * SQ + (SQ - cw) / 2, BOARD_Y + BOARD_S + 6),
                  c, font=coord_font, fill=TEXT_DIM)
    for rank in range(8):
        draw.text((BOARD_X - 16, BOARD_Y + (7 - rank) * SQ + SQ / 2 - 8),
                  str(rank + 1), font=coord_font, fill=TEXT_DIM)


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

    # Flat, print-style arrows: opacity encodes probability, the chosen
    # move is fully opaque in the vermillion accent.
    width = max(4, int(SQ * (0.10 + 0.16 * prob)))
    alpha = 235 if emphasize else int(90 + 100 * prob)

    layer = Image.new('RGBA', overlay.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.line([tuple((x1, y1)), tuple(base)], fill=color + (alpha,), width=width)
    wing = perp * (width * 1.35)
    d.polygon([tuple(tip), tuple(base + wing), tuple(base - wing)],
              fill=color + (alpha,))
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


def section_heading(draw, x, y, width, label):
    """Small-caps style heading over a hairline rule."""
    draw.text((x, y), label, font=font(13, bold=True), fill=TEXT)
    draw.line([x, y + 22, x + width, y + 22], fill=HAIRLINE, width=1)
    return y + 32


def draw_panel(draw, ply_data, ply_index, total_plies, history, shown_value):
    y = BOARD_Y - 6
    y = section_heading(draw, PANEL_X, y, PANEL_W, "POLICY  π(a | s)")
    mover = "White" if ply_data['white_to_move'] else "Black"
    draw.text((PANEL_X + PANEL_W - 118, BOARD_Y - 6),
              f"ply {ply_index + 1}/{total_plies}", font=font(13), fill=TEXT_DIM)
    draw.text((PANEL_X, y), f"{mover} to move · sampled at T = 1.2",
              font=font(13), fill=TEXT_DIM)
    y += 28

    # Candidate probability bars (thin marks on a light track)
    bar_w = PANEL_W - 130
    p_max = max(p for _, p in ply_data['top']) or 1.0
    for uci, p in ply_data['top']:
        chosen = uci == ply_data['move']
        draw.text((PANEL_X, y), uci, font=mono(14, bold=chosen),
                  fill=TEXT if chosen else TEXT_DIM)
        bx = PANEL_X + 72
        draw.rounded_rectangle([bx, y + 4, bx + bar_w, y + 14], radius=4,
                               fill=BAR_TRACK)
        fill_w = max(6, int(bar_w * 0.9 * p / p_max))
        draw.rounded_rectangle([bx, y + 4, bx + fill_w, y + 14], radius=4,
                               fill=ACCENT_WARM if chosen else ACCENT)
        draw.text((bx + bar_w + 12, y), f"{p * 100:4.1f}%", font=font(13),
                  fill=TEXT if chosen else TEXT_DIM)
        y += 25

    # Value head: diverging bar around a neutral midpoint
    y += 14
    y = section_heading(draw, PANEL_X, y, PANEL_W, "VALUE  v(s)")
    vx, vw = PANEL_X, PANEL_W - 78
    draw.rounded_rectangle([vx, y + 4, vx + vw, y + 14], radius=4, fill=BAR_TRACK)
    mid = vx + vw // 2
    draw.line([mid, y + 1, mid, y + 17], fill=TEXT_DIM, width=1)
    v = max(-1.0, min(1.0, shown_value))
    end = mid + int((vw // 2 - 2) * v)
    draw.rounded_rectangle([min(mid, end), y + 4, max(mid, end), y + 14],
                           radius=4, fill=ACCENT if v >= 0 else ACCENT_WARM)
    draw.text((vx + vw + 12, y - 1), f"{shown_value:+.2f}",
              font=mono(14, bold=True), fill=TEXT)
    draw.text((vx, y + 20), "−1 (Black)", font=font(11), fill=TEXT_DIM)
    lbl = "+1 (White)"
    draw.text((vx + vw - draw.textlength(lbl, font=font(11)), y + 20), lbl,
              font=font(11), fill=TEXT_DIM)
    y += 44

    # Move history
    y = section_heading(draw, PANEL_X, y, PANEL_W, "MOVES")
    recent = history[-16:]
    start_no = len(history) - len(recent)
    col, row = 0, 0
    for i, mv in enumerate(recent):
        n = start_no + i
        label = f"{n // 2 + 1}.{'' if n % 2 == 0 else '..'}{mv}"
        tx = PANEL_X + col * ((PANEL_W // 2) + 6)
        draw.text((tx, y + row * 21), label,
                  font=mono(13, bold=(i == len(recent) - 1)),
                  fill=TEXT if i == len(recent) - 1 else TEXT_DIM)
        col += 1
        if col == 2:
            col, row = 0, row + 1


def draw_eval_gauge(img, draw, shown_value):
    """Vertical white/black advantage gauge beside the board."""
    gy, gh, gw = BOARD_Y, BOARD_S, 18
    v = max(-1.0, min(1.0, shown_value))
    white_frac = 0.5 + v / 2
    split = gy + gh - int(gh * white_frac)
    draw.rectangle([GAUGE_X, gy, GAUGE_X + gw, split], fill=(70, 70, 70))
    draw.rectangle([GAUGE_X, split, GAUGE_X + gw, gy + gh], fill=(252, 252, 252))
    draw.rectangle([GAUGE_X, gy, GAUGE_X + gw, gy + gh],
                   outline=BOARD_EDGE, width=1)
    draw.line([GAUGE_X, gy + gh // 2, GAUGE_X + gw, gy + gh // 2],
              fill=(150, 150, 150), width=1)


def render_frame(bg, pieces, board_arr, *, last_move=None, moving=None,
                 fade_sq=None, fade_alpha=255, arrows=None, check_sq=None,
                 panel=None):
    img = bg.copy()
    draw = ImageDraw.Draw(img, 'RGBA')

    # Header, styled like a paper figure heading
    title_font = serif(24, bold=True)
    draw.text((BOARD_X - 16, 22), "Chess-RL: Self-Play with Policy and Value Heads",
              font=title_font, fill=TEXT)
    tag = "AlphaZero-style training loop"
    draw.text((W - 40 - draw.textlength(tag, font=font(13)), 30), tag,
              font=font(13), fill=TEXT_DIM)
    draw.line([BOARD_X - 16, 62, W - 40, 62], fill=HAIRLINE, width=1)

    # Figure caption in the lower right column
    cap1 = "Figure 1.  One self-play game. Arrows show the top-5 policy candidates"
    cap2 = "π(a | s) (opacity ∝ probability); the sampled move is drawn in vermillion."
    draw.text((PANEL_X, H - 62), cap1, font=serif(13), fill=TEXT_DIM)
    draw.text((PANEL_X, H - 42), cap2, font=serif(13), fill=TEXT_DIM)

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
        shadow.paste((0, 0, 0, 50), (0, 0), piece_img)
        img.paste(shadow, (x + 2, y + 3), shadow)
        img.paste(piece_img, (x, y), piece_img)

    if panel:
        draw_eval_gauge(img, draw, panel['shown_value'])
        draw_panel(draw, panel['ply'], panel['index'], panel['total'],
                   panel['history'], panel['shown_value'])

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

    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("ffmpeg failed")
    size = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out} ({size:.1f} MB)")


if __name__ == '__main__':
    main()
