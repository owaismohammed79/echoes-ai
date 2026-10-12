import { useEffect, useRef, useState } from "react";

import { cn } from "@/lib/utils";

/**
 * The mascot: a face that turns to watch the cursor.
 *
 * The artwork is a set of nine pre-drawn SVGs — eight compass facings plus a
 * neutral "front" — so the whole sprite is swapped on a direction change rather
 * than re-drawn per frame. That is what keeps it cheap: the pointer maths runs
 * on a single rAF tick, and React only re-renders when the facing actually
 * changes, never while the cursor moves within one sector.
 */

/** The eight sectors, ordered clockwise from north. */
export const COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"] as const;

export type Compass = (typeof COMPASS)[number];
/** `FRONT` is the resting pose, shown before the cursor is seen. */
export type Facing = Compass | "FRONT";

/** Slack, in px, around the sprite that still counts as "on" the mascot. */
const EDGE = 10;

/** One sprite per facing, served from `public/assets/mascot`. */
const SPRITES: Record<Facing, string> = {
  FRONT: "/assets/mascot/front.svg",
  N: "/assets/mascot/up.svg",
  NE: "/assets/mascot/top-right.svg",
  E: "/assets/mascot/right.svg",
  SE: "/assets/mascot/bottom-right.svg",
  S: "/assets/mascot/down.svg",
  SW: "/assets/mascot/bottom-left.svg",
  W: "/assets/mascot/left.svg",
  NW: "/assets/mascot/top-left.svg",
};

/**
 * Which way the mascot faces for a cursor at (dx, dy) from its centre.
 *
 * The angle is measured from north, clockwise, so it reads as a screen bearing:
 * 0deg is up, 90deg is right. Rounding to the nearest 45deg bucket folds the
 * full circle into the eight sectors of `COMPASS`, which is what makes the
 * sprite change at sector boundaries instead of continuously.
 */
export function facingFor(dx: number, dy: number): Facing {
  if (dx === 0 && dy === 0) return "FRONT";
  const deg = (Math.atan2(dx, -dy) * 180) / Math.PI;
  const sector = Math.round(((deg + 360) % 360) / 45) % 8;
  return COMPASS[sector];
}

export function MouseMascot({ className }: { className?: string }) {
  const [facing, setFacing] = useState<Facing>("FRONT");
  const ref = useRef<HTMLDivElement>(null);
  // Measured, not read per event: calling getBoundingClientRect on every
  // pointermove would force layout dozens of times a second for a value that
  // only changes on resize or scroll.
  const box = useRef<{
    cx: number;
    cy: number;
    halfW: number;
    halfH: number;
  } | null>(null);

  useEffect(() => {
    const measure = () => {
      const el = ref.current;
      if (!el) return;
      const rect = el.getBoundingClientRect();
      box.current = {
        cx: rect.left + rect.width / 2,
        cy: rect.top + rect.height / 2,
        halfW: rect.width / 2,
        halfH: rect.height / 2,
      };
    };
    measure();

    let frame = 0;
    const onPointerMove = (event: PointerEvent) => {
      if (frame) return;
      // At most one state update per animation frame.
      frame = window.requestAnimationFrame(() => {
        frame = 0;
        const b = box.current;
        if (!b) return;
        const dx = event.clientX - b.cx;
        const dy = event.clientY - b.cy;
        // A dead-zone over the sprite itself: a cursor on the mascot has no
        // meaningful bearing, and facing the viewer is the right answer there.
        const inside =
          Math.abs(dx) <= b.halfW + EDGE && Math.abs(dy) <= b.halfH + EDGE;
        const next = inside ? "FRONT" : facingFor(dx, dy);
        setFacing((prev) => (prev === next ? prev : next));
      });
    };
    // Leaving the window fires no move event, so watch for a pointerout with no
    // related target: that is the cursor crossing the document edge.
    const onPointerOut = (event: PointerEvent) => {
      if (!event.relatedTarget) setFacing("FRONT");
    };
    const onLeave = () => setFacing("FRONT");

    window.addEventListener("pointermove", onPointerMove, { passive: true });
    window.addEventListener("pointerout", onPointerOut);
    window.addEventListener("resize", measure);
    window.addEventListener("scroll", measure, { passive: true });
    document.addEventListener("mouseleave", onLeave);

    // Warm the other sprites once idle, so a sector change swaps instantly
    // rather than flashing an unloaded image the first time it is needed.
    const warm = window.setTimeout(() => {
      for (const src of Object.values(SPRITES)) {
        const img = new Image();
        img.src = src;
      }
    }, 1200);

    return () => {
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerout", onPointerOut);
      window.removeEventListener("resize", measure);
      window.removeEventListener("scroll", measure);
      document.removeEventListener("mouseleave", onLeave);
      window.clearTimeout(warm);
      if (frame) window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <div
      ref={ref}
      aria-hidden="true"
      data-facing={facing}
      // Deliberately chrome-less: no surface and no border, so the artwork sits
      // directly on the page rather than in a framed tile. The box exists only
      // to give the pointer maths a stable centre and to reserve layout space.
      className={cn("relative h-24 w-24 shrink-0", className)}
    >
      <img
        src={SPRITES[facing]}
        alt=""
        draggable={false}
        className="pointer-events-none absolute inset-0 h-full w-full select-none object-contain"
      />
    </div>
  );
}
