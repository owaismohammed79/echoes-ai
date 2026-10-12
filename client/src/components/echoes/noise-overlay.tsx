/**
 * Grain. Aceternity's Noise Texture, inlined: the registry does not expose it
 * as a public item (the CLI gets a 401), so this is the same technique — an
 * SVG feTurbulence tile at a few percent opacity, mixed over the surface.
 *
 * It carries no colour and casts no light; it exists so the glass drawer reads
 * as a material against flat black rather than as a rectangle of alpha.
 */
export function NoiseOverlay({ opacity = 0.035 }: { opacity?: number }) {
  return (
    <svg
      aria-hidden="true"
      className="pointer-events-none absolute inset-0 h-full w-full mix-blend-overlay"
      style={{ opacity }}
    >
      <filter id="echoes-grain">
        <feTurbulence
          type="fractalNoise"
          baseFrequency="0.8"
          numOctaves="3"
          stitchTiles="stitch"
        />
      </filter>
      <rect width="100%" height="100%" filter="url(#echoes-grain)" />
    </svg>
  );
}
