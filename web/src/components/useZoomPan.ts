import { useCallback, useEffect, useRef, useState } from "react";

export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/** Zoom + pan for an SVG editor. Wheel zooms toward the cursor, buttons zoom
 *  about the centre, drag pans. Zoom is clamped to [fit, 40×]. The caller owns
 *  clicks/drags — panMove reports whether it actually panned (moved > 3px) so a
 *  plain click is never eaten. Works with preserveAspectRatio="xMidYMid meet". */
export function useZoomPan(base: Box) {
  const [view, setView] = useState<Box>(base);
  const baseRef = useRef(base);
  baseRef.current = base;
  const viewRef = useRef(view);
  viewRef.current = view;
  const svgRef = useRef<SVGSVGElement>(null);
  const pan = useRef<{ sx: number; sy: number; px: number; py: number; moved: boolean } | null>(null);

  const toLocal = useCallback((clientX: number, clientY: number): [number, number] => {
    const svg = svgRef.current!;
    const p = svg.createSVGPoint();
    p.x = clientX;
    p.y = clientY;
    const d = p.matrixTransform(svg.getScreenCTM()!.inverse());
    return [d.x, d.y];
  }, []);

  const zoomAt = useCallback((factor: number, cx: number, cy: number) => {
    setView((v) => {
      const minW = baseRef.current.w / 40;
      const maxW = baseRef.current.w;
      let f = factor;
      if (v.w * f > maxW) f = maxW / v.w;
      if (v.w * f < minW) f = minW / v.w;
      return { x: cx - (cx - v.x) * f, y: cy - (cy - v.y) * f, w: v.w * f, h: v.h * f };
    });
  }, []);

  // native, non-passive wheel listener so preventDefault stops page scroll
  useEffect(() => {
    const svg = svgRef.current;
    if (!svg) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const [cx, cy] = toLocal(e.clientX, e.clientY);
      zoomAt(e.deltaY > 0 ? 1.15 : 1 / 1.15, cx, cy);
    };
    svg.addEventListener("wheel", onWheel, { passive: false });
    return () => svg.removeEventListener("wheel", onWheel);
  }, [toLocal, zoomAt]);

  const zoomButton = useCallback((factor: number) => {
    const v = viewRef.current;
    zoomAt(factor, v.x + v.w / 2, v.y + v.h / 2);
  }, [zoomAt]);

  const fit = useCallback(() => setView(baseRef.current), []);

  const panStart = useCallback((clientX: number, clientY: number) => {
    pan.current = { sx: clientX, sy: clientY, px: clientX, py: clientY, moved: false };
  }, []);

  const panMove = useCallback((clientX: number, clientY: number): boolean => {
    if (!pan.current) return false;
    if (Math.abs(clientX - pan.current.sx) > 3 || Math.abs(clientY - pan.current.sy) > 3)
      pan.current.moved = true;
    if (!pan.current.moved) return false;
    const svg = svgRef.current!;
    const rect = svg.getBoundingClientRect();
    const v = viewRef.current;
    const scale = Math.max(v.w / rect.width, v.h / rect.height); // data units per screen px
    const dx = (clientX - pan.current.px) * scale;
    const dy = (clientY - pan.current.py) * scale;
    setView((cur) => ({ ...cur, x: cur.x - dx, y: cur.y - dy }));
    pan.current.px = clientX;
    pan.current.py = clientY;
    return true;
  }, []);

  const panEnd = useCallback((): boolean => {
    const moved = pan.current?.moved ?? false;
    pan.current = null;
    return moved;
  }, []);

  const viewBox = `${view.x} ${view.y} ${view.w} ${view.h}`;
  const zoomFactor = base.w / view.w;
  return { svgRef, viewBox, view, zoomFactor, zoomButton, fit, panStart, panMove, panEnd };
}
