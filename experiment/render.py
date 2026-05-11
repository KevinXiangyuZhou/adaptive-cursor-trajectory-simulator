"""
Shared pygame rendering functions for hcs_package experiments.

These functions take explicit parameters (not an env object) and are
designed for trajectory-replay workflows.
"""
import pygame
from typing import List, Tuple, Optional, Dict

from experiment.utils import generateTunnelBoundaries

SCALE = 1000  # display pixels per metre


def m_to_px(val: float, scale: float = SCALE) -> int:
    """Convert metres to display pixels."""
    return int(val * scale)


def render_tunnel_frame(
    screen: pygame.Surface,
    tunnel_path_m: List[Tuple[float, float]],
    tunnel_width_m: float,
    cursor_pos_px: Tuple[int, int],
    target_pos_m: Tuple[float, float],
    target_radius_m: float,
    trail_px: Optional[List[Tuple[int, int]]] = None,
    scale: float = SCALE,
    left_boundary_px: Optional[List[Tuple[int, int]]] = None,
    right_boundary_px: Optional[List[Tuple[int, int]]] = None,
):
    """
    Render a single frame of a tunnel/steering/turning task.

    Args:
        screen: Pygame screen surface
        tunnel_path_m: Tunnel centerline in metres [(x, y), ...]
        tunnel_width_m: Tunnel width in metres
        cursor_pos_px: Current cursor position in display pixels (x, y)
        target_pos_m: Target position in metres (x, y)
        target_radius_m: Target radius in metres
        trail_px: List of past cursor positions in display pixels
        scale: Display scale (pixels per metre)
        left_boundary_px: Pre-computed left boundary in pixels (optional)
        right_boundary_px: Pre-computed right boundary in pixels (optional)
    """
    display_w, display_h = screen.get_size()

    # Compute boundaries if not provided
    if left_boundary_px is None or right_boundary_px is None:
        left_m, right_m = generateTunnelBoundaries(tunnel_path_m, tunnel_width_m)
        left_boundary_px = [(m_to_px(x, scale), m_to_px(y, scale)) for x, y in left_m]
        right_boundary_px = [(m_to_px(x, scale), m_to_px(y, scale)) for x, y in right_m]

    # Background
    screen.fill((255, 255, 255))

    # Shaded overlay outside tunnel
    overlay = pygame.Surface((display_w, display_h), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    poly = left_boundary_px + right_boundary_px[::-1]
    if len(poly) >= 3:
        pygame.draw.polygon(overlay, (255, 255, 255, 0), poly)
    screen.blit(overlay, (0, 0))

    # Tunnel boundaries (blue lines)
    if len(left_boundary_px) >= 2:
        pygame.draw.lines(screen, (0, 0, 255), False, left_boundary_px, 2)
    if len(right_boundary_px) >= 2:
        pygame.draw.lines(screen, (0, 0, 255), False, right_boundary_px, 2)

    # Target (red circle)
    target_px = (m_to_px(target_pos_m[0], scale), m_to_px(target_pos_m[1], scale))
    target_r_px = max(1, m_to_px(target_radius_m, scale))
    pygame.draw.circle(screen, (255, 0, 0), target_px, target_r_px)

    # Trajectory trail (black line)
    if trail_px and len(trail_px) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, trail_px, 1)

    # Cursor (black dot)
    cursor_r = max(1, int(0.005 * scale))
    pygame.draw.circle(screen, (0, 0, 0), cursor_pos_px, cursor_r)


def render_lasso_frame(
    screen: pygame.Surface,
    icons: List[Dict],
    cursor_pos_px: Tuple[int, int],
    trail_px: Optional[List[Tuple[int, int]]] = None,
    start_pos_m: Optional[Tuple[float, float]] = None,
    end_pos_m: Optional[Tuple[float, float]] = None,
    end_radius_m: float = 0.004,
    scale: float = SCALE,
    icon_rects: Optional[List[Tuple[Dict, pygame.Rect]]] = None,
):
    """
    Render a single frame of a lasso selection task.

    Args:
        screen: Pygame screen surface
        icons: List of icon dicts with keys: x, y, radius (metres), is_target, is_empty
        cursor_pos_px: Current cursor position in display pixels (x, y)
        trail_px: List of past cursor positions in display pixels
        start_pos_m: Start marker position in metres (optional)
        end_pos_m: End marker position in metres (optional)
        end_radius_m: End target radius in metres
        scale: Display scale (pixels per metre)
        icon_rects: Pre-computed icon rects as [(icon_dict, pygame.Rect), ...] (optional)
    """
    # Background
    screen.fill((255, 255, 255))

    # Compute icon rects if not provided
    if icon_rects is None:
        icon_rects = []
        for icon in icons:
            r_px = max(1, m_to_px(icon['radius'], scale))
            ix = m_to_px(icon['x'], scale)
            iy = m_to_px(icon['y'], scale)
            icon_rects.append((icon, pygame.Rect(ix - r_px, iy - r_px, r_px * 2, r_px * 2)))

    # Draw icons
    for icon, rect in icon_rects:
        if icon.get('is_empty', False):
            color = (211, 211, 211)  # Light grey
        elif icon.get('is_target', False):
            color = (255, 165, 0)    # Orange
        else:
            color = (128, 128, 128)  # Grey
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, (0, 0, 0), rect, 2)

    # Start marker (green)
    if start_pos_m:
        start_px = (m_to_px(start_pos_m[0], scale), m_to_px(start_pos_m[1], scale))
        pygame.draw.circle(screen, (0, 255, 0), start_px, max(1, int(0.005 * scale)))

    # End marker (red with radius circle)
    if end_pos_m:
        end_px = (m_to_px(end_pos_m[0], scale), m_to_px(end_pos_m[1], scale))
        end_r_px = max(1, m_to_px(end_radius_m, scale))
        pygame.draw.circle(screen, (255, 0, 0), end_px, max(1, int(0.005 * scale)))
        pygame.draw.circle(screen, (255, 0, 0), end_px, end_r_px, 1)

    # Trajectory trail (green line)
    if trail_px and len(trail_px) > 1:
        pygame.draw.lines(screen, (0, 200, 0), False, trail_px, 1)

    # Cursor (black dot)
    cursor_r = max(1, int(0.005 * scale))
    pygame.draw.circle(screen, (0, 0, 0), cursor_pos_px, cursor_r)


def precompute_tunnel_boundaries(
    tunnel_path_m: List[Tuple[float, float]],
    tunnel_width_m: float,
    scale: float = SCALE
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Pre-compute tunnel boundaries in display pixels.

    Call once before the render loop to avoid recomputing every frame.

    Returns:
        (left_boundary_px, right_boundary_px)
    """
    left_m, right_m = generateTunnelBoundaries(tunnel_path_m, tunnel_width_m)
    left_px = [(m_to_px(x, scale), m_to_px(y, scale)) for x, y in left_m]
    right_px = [(m_to_px(x, scale), m_to_px(y, scale)) for x, y in right_m]
    return left_px, right_px


def precompute_icon_rects(
    icons: List[Dict],
    scale: float = SCALE
) -> List[Tuple[Dict, pygame.Rect]]:
    """
    Pre-compute icon rects in display pixels.

    Call once before the render loop to avoid recomputing every frame.

    Args:
        icons: List of icon dicts with keys: x, y, radius (metres)

    Returns:
        List of (icon_dict, pygame.Rect) tuples
    """
    rects = []
    for icon in icons:
        r_px = max(1, m_to_px(icon['radius'], scale))
        ix = m_to_px(icon['x'], scale)
        iy = m_to_px(icon['y'], scale)
        rects.append((icon, pygame.Rect(ix - r_px, iy - r_px, r_px * 2, r_px * 2)))
    return rects


def render_menu_frame(
    screen: pygame.Surface,
    main_menu_rect: Dict,
    sub_menu_rect: Dict,
    main_menu_items: List[Dict],
    sub_menu_items: List[Dict],
    cursor_pos_m: Tuple[float, float],
    cursor_pos_px: Tuple[int, int],
    target_pos_m: Tuple[float, float],
    target_radius_m: float,
    trail_px: Optional[List[Tuple[int, int]]] = None,
    scale: float = SCALE,
):
    """
    Render a single frame of a cascading menu task (macOS-style).

    Args:
        screen: Pygame screen surface
        main_menu_rect: Dict with x, y, width, height (metres)
        sub_menu_rect: Dict with x, y, width, height (metres)
        main_menu_items: List of item dicts (index, x, y, width, height, is_target)
        sub_menu_items: List of item dicts
        cursor_pos_m: Cursor position in metres (for hover detection)
        cursor_pos_px: Cursor position in display pixels
        target_pos_m: Target position in metres
        target_radius_m: Target radius in metres
        trail_px: List of past cursor positions in display pixels
        scale: Display scale (pixels per metre)
    """
    # Colors (matching original macOS-style)
    menu_bg_color = (255, 255, 255)  # White
    menu_border_color = (192, 192, 192)  # Light gray
    shadow_color = (0, 0, 0, 38)  # rgba with alpha
    target_color = (255, 107, 107)  # Light red #FF6B6B
    hover_color = (229, 229, 229)  # Light gray #E5E5E5
    text_color = (0, 0, 0)
    text_color_white = (255, 255, 255)

    # Background
    screen.fill((255, 255, 255))

    cursor_x, cursor_y = cursor_pos_m
    shadow_offset = 3

    # -- Draw main menu --
    main_x_px = m_to_px(main_menu_rect['x'], scale)
    main_y_px = m_to_px(main_menu_rect['y'], scale)
    main_w_px = m_to_px(main_menu_rect['width'], scale)
    main_h_px = m_to_px(main_menu_rect['height'], scale)

    # Shadow
    shadow_surface = pygame.Surface((main_w_px + shadow_offset, main_h_px + shadow_offset), pygame.SRCALPHA)
    shadow_surface.fill(shadow_color)
    screen.blit(shadow_surface, (main_x_px + shadow_offset, main_y_px + shadow_offset))

    # Background
    main_rect_pg = pygame.Rect(main_x_px, main_y_px, main_w_px, main_h_px)
    pygame.draw.rect(screen, menu_bg_color, main_rect_pg)
    pygame.draw.rect(screen, menu_border_color, main_rect_pg, 1)

    # Font size based on item height
    if main_menu_items:
        first_item_h_px = m_to_px(main_menu_items[0]['height'], scale)
        font_size = max(10, int(first_item_h_px * 0.6))
    else:
        font_size = 12
    try:
        font = pygame.font.Font(None, font_size)
    except:
        font = pygame.font.SysFont('Arial', font_size)

    # Draw main menu items
    for item in main_menu_items:
        item_x_px = m_to_px(item['x'], scale)
        item_y_px = m_to_px(item['y'], scale)
        item_w_px = m_to_px(item['width'], scale)
        item_h_px = m_to_px(item['height'], scale)
        item_rect = pygame.Rect(item_x_px, item_y_px, item_w_px, item_h_px)

        is_hovered = (item['x'] <= cursor_x <= item['x'] + item['width'] and
                      item['y'] <= cursor_y <= item['y'] + item['height'])

        if item['is_target']:
            item_color = target_color
            text_col = text_color_white
        elif is_hovered:
            item_color = hover_color
            text_col = text_color
        else:
            item_color = None
            text_col = text_color

        if item_color:
            pygame.draw.rect(screen, item_color, item_rect)

        label = f"Item {item['index'] + 1}"
        text_surface = font.render(label, True, text_col)
        text_rect = text_surface.get_rect(center=(item_x_px + item_w_px // 2, item_y_px + item_h_px // 2))
        if text_rect.left >= item_x_px and text_rect.right <= item_x_px + item_w_px:
            screen.blit(text_surface, text_rect)

        # Arrow indicator on target item
        if item['is_target']:
            arrow_size = max(3, int(item_h_px * 0.15))
            arrow_x = item_x_px + item_w_px - arrow_size - 2
            arrow_y = item_y_px + item_h_px // 2
            arrow_points = [
                (arrow_x, arrow_y - arrow_size // 2),
                (arrow_x, arrow_y + arrow_size // 2),
                (arrow_x + arrow_size, arrow_y)
            ]
            pygame.draw.polygon(screen, text_color_white, arrow_points)

    # Redraw main menu right border
    pygame.draw.line(screen, menu_border_color,
                     (main_x_px + main_w_px, main_y_px),
                     (main_x_px + main_w_px, main_y_px + main_h_px), 1)

    # -- Draw submenu --
    sub_x_px = m_to_px(sub_menu_rect['x'], scale)
    sub_y_px = m_to_px(sub_menu_rect['y'], scale)
    sub_w_px = m_to_px(sub_menu_rect['width'], scale)
    sub_h_px = m_to_px(sub_menu_rect['height'], scale)

    shadow_surface = pygame.Surface((sub_w_px + shadow_offset, sub_h_px + shadow_offset), pygame.SRCALPHA)
    shadow_surface.fill(shadow_color)
    screen.blit(shadow_surface, (sub_x_px + shadow_offset, sub_y_px + shadow_offset))

    sub_rect_pg = pygame.Rect(sub_x_px, sub_y_px, sub_w_px, sub_h_px)
    pygame.draw.rect(screen, menu_bg_color, sub_rect_pg)
    pygame.draw.rect(screen, menu_border_color, sub_rect_pg, 1)

    for item in sub_menu_items:
        item_x_px = m_to_px(item['x'], scale)
        item_y_px = m_to_px(item['y'], scale)
        item_w_px = m_to_px(item['width'], scale)
        item_h_px = m_to_px(item['height'], scale)
        item_rect = pygame.Rect(item_x_px, item_y_px, item_w_px, item_h_px)

        is_hovered = (item['x'] <= cursor_x <= item['x'] + item['width'] and
                      item['y'] <= cursor_y <= item['y'] + item['height'])

        if item['is_target']:
            item_color = target_color
            text_col = text_color_white
        elif is_hovered:
            item_color = hover_color
            text_col = text_color
        else:
            item_color = None
            text_col = text_color

        if item_color:
            pygame.draw.rect(screen, item_color, item_rect)

        label = f"Sub {item['index'] + 1}"
        text_surface = font.render(label, True, text_col)
        text_rect = text_surface.get_rect(center=(item_x_px + item_w_px // 2, item_y_px + item_h_px // 2))
        if text_rect.left >= item_x_px and text_rect.right <= item_x_px + item_w_px:
            screen.blit(text_surface, text_rect)

    # Trail
    if trail_px and len(trail_px) > 1:
        pygame.draw.lines(screen, (0, 0, 0), False, trail_px, 1)

    # Cursor (black dot)
    cursor_r = max(1, int(0.005 * scale))
    pygame.draw.circle(screen, (0, 0, 0), cursor_pos_px, cursor_r)
