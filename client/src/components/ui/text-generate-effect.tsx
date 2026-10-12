"use client";
import { useEffect } from "react";
import { motion, stagger, useAnimate } from "motion/react";

/**
 * Aceternity's Text Generate Effect, adapted in exactly one place.
 *
 * Upstream reveals words on a fixed `stagger(0.2)` timer, which is the wrong
 * base for a transcript: the text is supposed to *be* the audio, not a slideshow
 * running beside it. So an optional `revealed` prop takes over — when it is a
 * number, word N is shown once `revealed > N`, and the caller drives that number
 * from the engine's real word-boundary events. Omit the prop and this behaves
 * exactly as upstream, timer and all.
 *
 * Two other departures, both required by the design system:
 *   - Upstream's own `font-bold text-2xl mt-4` is gone. Hierarchy here comes
 *     from size and tracking, never weight, and the call site owns typography.
 *   - The blur is upstream's, kept because it is what makes a word read as
 *     *arriving* rather than snapping on. It is damped from 10px to 6px so a
 *     long sentence is not animating a heavy filter 30 times over.
 *
 * Every word stays in the DOM and only its opacity changes, so an enclosing
 * aria-live region announces the whole sentence once instead of muttering it
 * out word by word.
 */
export const TextGenerateEffect = ({
  words,
  className,
  filter = true,
  duration = 0.5,
  revealed,
}: {
  words: string;
  className?: string;
  filter?: boolean;
  duration?: number;
  /** Words shown so far. Omit to keep upstream's staggered timer. */
  revealed?: number;
}) => {
  const [scope, animate] = useAnimate();
  const wordsArray = words.split(" ");
  const audioDriven = revealed !== undefined;

  useEffect(() => {
    // In audio-driven mode the reveal is a prop, so there is no timer to run.
    if (audioDriven) return;
    animate(
      "span",
      {
        opacity: 1,
        filter: filter ? "blur(0px)" : "none",
      },
      {
        duration: duration ? duration : 1,
        delay: stagger(0.2),
      }
    );
  }, [scope.current, audioDriven, animate, duration, filter]);

  return (
    <motion.span ref={scope} className={className}>
      {wordsArray.map((word, idx) => {
        const on = idx < (revealed ?? 0);
        return (
          <motion.span
            key={`${word}-${idx}`}
            // Nothing is hidden from assistive tech, only from the eye.
            className={audioDriven ? undefined : "opacity-0"}
            style={
              audioDriven
                ? undefined
                : { filter: filter ? "blur(6px)" : "none" }
            }
            animate={
              audioDriven
                ? {
                    opacity: on ? 1 : 0,
                    filter: filter ? (on ? "blur(0px)" : "blur(6px)") : "none",
                  }
                : undefined
            }
            // Falling back to zero keeps a re-reveal (new sentence, or a
            // resumed segment) from cross-fading the previous words out.
            transition={{ duration: on ? duration : 0, ease: "easeOut" }}
          >
            {word}{" "}
          </motion.span>
        );
      })}
    </motion.span>
  );
};
