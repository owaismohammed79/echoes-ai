import { useState } from "react";

import { cn } from "@/lib/utils";

/**
 * A landmark's source thumbnail, with a topographic stand-in.
 *
 * Wikipedia's pageimages are the one piece of a citation that shows rather than
 * tells, but they are also the one piece that can be absent (a page with no
 * lead image) or dead (a hotlinked upload URL that has moved). Both cases fall
 * back to the same contour mark, so a row never collapses to a broken-image
 * glyph and the grid stays even.
 */

/** Contour lines, drawn in `currentColor` so the caller sets the tone. */
function TopographicMark({ className }: { className?: string }) {
  return (
    <svg
      viewBox="0 0 48 48"
      fill="none"
      aria-hidden="true"
      className={className}
    >
      <path
        d="M4 26c2-11 11-19 22-19 10 0 18 8 18 19 0 11-10 18-20 18C13 44 6 37 4 26Z"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.7"
      />
      <path
        d="M11 27c1-8 8-14 16-14s14 6 14 14-7 13-15 13c-9 0-15-6-15-13Z"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.85"
      />
      <path
        d="M17 28c1-6 6-10 11-10s10 4 10 10-5 9-10 9c-6 0-11-4-11-9Z"
        stroke="currentColor"
        strokeWidth="1"
      />
      <path
        d="M23 28c0-3 2-5 5-5s4 2 4 5-2 4-4 4c-3 0-5-2-5-4Z"
        stroke="currentColor"
        strokeWidth="1"
        opacity="0.7"
      />
    </svg>
  );
}

export function LandmarkThumbnail({
  src,
  alt,
  className,
}: {
  /** `fact.imageUrl` — the provider thumbnail, when there was one. */
  src?: string;
  /** The landmark name, for the image's accessible label. */
  alt: string;
  /** Sizing/placement only; the frame, radius and fallback live here. */
  className?: string;
}) {
  const [failed, setFailed] = useState(false);
  const showImage = Boolean(src) && !failed;

  return (
    <div
      className={cn(
        "relative h-12 w-12 shrink-0 overflow-hidden rounded-sm border border-white/10 bg-[#101010]",
        className
      )}
    >
      {showImage ? (
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          draggable={false}
          onError={() => setFailed(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <TopographicMark className="h-full w-full text-[#10B981]/35" />
      )}
    </div>
  );
}
