import type { Poly } from "../api";

/** The signature moment: the traced tool as a knockout hero silhouette on a
 *  field ground — the "oversized mass" of the 2046 poster grammar. Optionally
 *  overlays the pocket outline (tool + clearance) as a hairline in the spot. */
export function Silhouette({
  tool,
  pocket,
  ground = "field",
  fill = "knockout",
}: {
  tool: Poly;
  pocket?: Poly | null;
  ground?: string;
  fill?: string;
}) {
  const rings = [tool.exterior, ...tool.holes, ...(pocket ? [pocket.exterior] : [])];
  const xs = rings.flatMap((r) => r.map((p) => p[0]));
  const ys = rings.flatMap((r) => r.map((p) => p[1]));
  const pad = 6;
  const minx = Math.min(...xs) - pad;
  const miny = Math.min(...ys) - pad;
  const w = Math.max(...xs) - minx + pad;
  const h = Math.max(...ys) - miny + pad;
  // bin frame is y-up; SVG is y-down → flip within the viewBox
  const path = (rings2: Poly) =>
    [rings2.exterior, ...rings2.holes]
      .map(
        (ring) =>
          "M " +
          ring.map(([x, y]) => `${x - minx} ${h - (y - miny)}`).join(" L ") +
          " Z",
      )
      .join(" ");

  const bg = `var(--c-${ground})`;
  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      className="w-full h-full"
      style={{ background: bg }}
      role="img"
      aria-label="Traced tool silhouette"
    >
      {pocket && (
        <path
          d={path(pocket)}
          fill="none"
          stroke="var(--c-teal)"
          strokeWidth={Math.max(w, h) / 400}
          strokeDasharray={`${Math.max(w, h) / 120} ${Math.max(w, h) / 200}`}
        />
      )}
      <path d={path(tool)} fill={`var(--c-${fill})`} fillRule="evenodd" />
    </svg>
  );
}
