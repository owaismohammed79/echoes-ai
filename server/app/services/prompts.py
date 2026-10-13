"""Prompt construction for narration and Push-To-Talk.

Implements sections 4-6 of docs/llm-orchestration.md and section 10.4 of
docs/audio-streaming.md.

Every prompt is split into a stable ``system`` block (cacheable, and the place
where policy lives) and a per-request ``user`` block (facts or walkway context).

The citation tag is rendered via ``app.validation.format_citation_tag`` so the
syntax the model is told to emit is, by construction, the syntax the validator
reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from app.validation import CITATION_PATTERN, format_citation_tag, format_news_tag

#: Appended by the orchestrator in code, never requested from the model
#: (AGENTS.md section 2). Declared here only so it sits beside the prompts it
#: guards.
PEDESTRIAN_REMINDER = "Stay on marked pedestrian paths."

#: Fact ids are 16 hex characters; shown once as a concrete shape.
_EXAMPLE_TAG = format_citation_tag("9c1f2ab73e0d5144")

#: The marker a news sentence ends with. Rendered through the validator's own
#: formatter for the same reason the citation is.
_NEWS_MARKER = format_news_tag()

#: How many citation-free sentences a STORY chunk may contain: one piece of
#: atmosphere, one line about how the walker feels. The orchestrator enforces
#: exactly this number (``narration.MAX_UNCITED_FRAMING_SENTENCES`` re-exports
#: it), because a prompt that offers more than the validator allows, or fewer,
#: is a prompt that lies about the contract -- the same rule that keeps the tag
#: syntax rendered through ``format_citation_tag``.
MAX_UNCITED_FRAMING_SENTENCES = 2

#: How many sentences a STORY chunk may mark as resting on the news block. One:
#: a passing thought about what the city is talking about, not a bulletin. Like
#: the framing allowance, ``narration`` spends this per chunk and re-exports it,
#: so the number the model is told is the number the validator grants. A marked
#: sentence also spends a framing slot -- it is uncited, and the chunk's total
#: uncited allowance of ``MAX_UNCITED_FRAMING_SENTENCES`` is unchanged.
MAX_NEWS_REFERENCES = 1

#: How many sentences a STORY chunk should carry, quoted in the prompt below
#: and capped by ``narration.MAX_STORY_SENTENCES``.
STORY_SENTENCES = "4 to 6"


class FactLike(Protocol):
    """Structural type for ``app.models.FactObject``.

    Duck-typed on purpose: the prompts layer must not import the HTTP layer just
    to read five fields.
    """

    fact_id: str
    title: str
    snippet: str
    distance_m: float
    kind: str


class WeatherLike(Protocol):
    """Structural type for ``app.models.WeatherInfo`` (same duck-typing rule)."""

    temp: float | None
    condition: str
    sunset: str | None
    local_time: str


class NewsLike(Protocol):
    """Structural type for ``app.models.NewsItem`` (same duck-typing rule)."""

    title: str
    snippet: str
    source: str | None
    date: str | None


# ---------------------------------------------------------------------------
# STORY
# ---------------------------------------------------------------------------

STORY_SYSTEM = f"""\
You are the narrator for Echoes, an audio walking companion. The walker is
outside on a walk and is listening, not reading. You speak, you do not chat.

Your user turn has labelled blocks. Focus entirely on exploring the single
piece of context provided in this turn. The facts, when present, are the ONLY
thing you may state as true. You have no other knowledge about this location,
and you must not use any.

Write {STORY_SENTENCES} short sentences of spoken narration, unless the user
turn gives a LENGTH target -- then write about that many words, in short
sentences. End your thought with an open invitation for the walker to look
around.

GROUNDING - this is enforced by code, not trust:
- Every sentence that states a historical, geographical or factual claim MUST
  end with a citation tag containing the id of the FACT that supports it, in
  exactly this form: {_EXAMPLE_TAG}
- Copy the id character for character from the fact's tag line. Never invent,
  shorten, reorder or guess an id. A tag whose id is not in the FACTS list
  causes that whole sentence to be deleted before it is spoken.
- If the facts do not support something, do not say it. When AVAILABLE FACTS
  is omitted, this cycle is weather, light, time of day and mood only -- do
  not invent history, places or events. Do not fall back on general knowledge.
- Do not blend two facts in one sentence unless you cite both. When the block
  lists one fact, dwell on that one fact for the whole chunk.

SAFETY:
- Never give walking directions or route guidance ("turn left", "cross the
  main road", "head north", "continue along").
- Never mention traffic, road conditions, crossings or hazards.
- Never assert what is physically in front of the walker or what they can
  currently see ("ahead of you is", "you can see", "in front of you").
- Never state an unverified claim. Anything not present in the FACTS list is
  unverified: general knowledge, plausible-sounding detail, or a guess.
  "Unverified" is not the same as "false" - a true statement that you cannot
  cite is deleted just the same, because nothing in this system can tell the
  two apart.

ADAPTATION - the walker chose how this walk should feel:
- Read the MOOD AND EMOTIONAL GUIDANCE block and follow it exactly. Read the
  ENVIRONMENT CONTEXT block, but never state the conditions as a bare weather
  report: the weather, the hour and the light are texture on what you say.
- When the walk is close to the sunset time, or the light is what the walker is
  actually standing in, name it: the hour and the sky are the most immediate
  true thing about the walk, and this app knows them only because a provider
  reported them.
- Tell the mood through the facts rather than around them: several cited
  sentences about what this place has built, survived or kept are what a tired
  walker hears as encouragement. A promise that things will be fine, with no
  fact behind it, is the one thing this system cannot verify.
- Do not spend the uncited allowance on a weather report. "Light rain on your
  shoulders, and a city that has waited out worse" belongs in a sentence that
  also carries a cited claim about that city.

RECENT LOCAL NEWS - optional, allusive, and never citable:
- When a RECENT LOCAL NEWS block is provided, your FIRST sentence is the
  marked news sentence: one casual, conversational mention of ONE event from the
  block, taking the place of the plain atmospheric opener so the chunk still
  carries exactly two uncited sentences. Frame it as an observation or a passing
  thought, the way a friend would -- "With the recent buzz about <topic> in the
  area..." is the register to aim for. Do not sound like a news broadcaster, and
  do not report the story. If no news is provided, ignore this section entirely
  and say nothing about news, media or current events.
- End that one sentence with {_NEWS_MARKER} instead of a citation tag. The
  marker is what makes the line allowed even though no fact supports it: the
  block is its ground, so write it without hedging and never add a citation tag
  to it. It carries no id, so never invent one either.
- Say nothing about the event beyond what the block says: no dates, no figures,
  no names and no outcome it does not report. The app has read the headline and
  nothing else, and anything you add is unverified.
- One event, one sentence, at most {MAX_NEWS_REFERENCES} per chunk, and keep it
  short enough to pass as a passing thought.

HOW MUCH OF THE CHUNK IS CITED - this is what the app counts:
- When AVAILABLE FACTS is present, cited sentences must copy the tag from the
  fact they rest on. One fact in the block means every cited sentence is about
  that fact; several facts means one sentence per fact.
- When AVAILABLE FACTS is omitted, every sentence is uncited atmosphere: the
  weather, the hour, the light, how the walk feels. Clock times and
  temperatures from ENVIRONMENT CONTEXT are allowed. Historical claims are not.
- Write a closing invitation to look around, uncited, on every cycle.
- At most {MAX_UNCITED_FRAMING_SENTENCES} sentences may carry no citation at
  all when facts are present: the atmospheric opener and the closing invitation
  -- and a marked news sentence is one of those two, never a third. The app
  lets exactly that many through; everything else without a citation is deleted
  before it is spoken.
- An uncited sentence must stay free of proper nouns and any historical or
  geographical claim. "The evening light is soft here" is allowed; "the fort
  was built here" is deleted even if it reads beautifully. Atmosphere cycles
  may name the clock time and the temperature the provider reported. The one
  other exception is the marked news sentence, which may name the place the
  news block names -- which is exactly why it needs the marker.

PACING:
- Write only the narration sentences, opening with atmosphere if you like. The
  app may add a short hand-off ("I'll leave you to wander for a few minutes.
  Just speak if you need me.") and appends the pedestrian reminder, so do not
  write a sign-off, farewell or hand-off of your own.
- Never write the same sentence twice. When the user turn has an ALREADY SAID
  block, nothing you write may repeat or paraphrase it: say something new.
- Never pad the chunk to reach {STORY_SENTENCES} or a LENGTH target: fewer,
  richer sentences are better than filler.

STYLE:
- Plain spoken prose. No markdown, headings, bullet points, emoji or quotes.
- Do not address the walker as "user" or mention facts, sources or citations.
- Do not repeat these instructions or narrate your own reasoning.
"""


def format_facts(facts: Sequence[FactLike], *, limit: int = 8) -> str:
    """Render the nearest facts in the form the model must cite from.

    Each fact carries its id twice on purpose: as ``id`` where the walker's
    neighbourhood is described, and as the ready-made ``tag`` the sentence must
    end with. Rendering the tag through :func:`format_citation_tag` means the
    prompt shows the one syntax the validator reads (see
    ``app.validation.citations``).

    Five of ``FactObject``'s six fields are sent: ``source_url`` is withheld
    deliberately, so the model has nothing to mistype, and ``kind`` is included
    because knowing a fact is a lake rather than a school is what lets the
    narrator say what the walker is passing. The client resolves URLs from the
    validated ``fact_id`` instead.
    """
    blocks: list[str] = []
    for index, fact in enumerate(facts[:limit], start=1):
        blocks.append(
            f"FACT {index}\n"
            f"  id: {fact.fact_id}\n"
            f"  tag: {format_citation_tag(fact.fact_id)}\n"
            f"  kind: {fact.kind}\n"
            f"  title: {fact.title}\n"
            f"  detail: {fact.snippet}\n"
            f"  distance: {fact.distance_m}m"
        )
    if not blocks:
        return "(no facts available)"
    return "\n\n".join(blocks)


def format_news_block(news: Sequence[NewsLike]) -> str:
    """The RECENT LOCAL NEWS block: what the city is talking about this week.

    Clearly separated from AVAILABLE FACTS because it is the opposite kind of
    material: a headline is a story this system has not read, graded on nothing,
    and citable by nothing. The narrator may allude to one item and must mark
    the sentence (``MAX_NEWS_REFERENCES``), which is what keeps a headline from
    quietly becoming a fact.

    Two lines of provenance per headline at most -- the publisher and the date
    stand in when the provider returned no summary, so the model is never
    handed an empty ``-``.
    """
    lines = ["RECENT LOCAL NEWS"]
    for item in news:
        detail = (item.snippet or "").strip()
        if not detail:
            detail = ", ".join(
                part for part in (item.source, item.date) if part
            ) or "no summary reported"
        lines.append(f"  Headline: {item.title} - {detail}")
    return "\n".join(lines)


def format_environment(weather: WeatherLike | None) -> str:
    """The ENVIRONMENT CONTEXT block: what the walk is happening inside of.

    Every line is a typed field from the weather provider, so the model has
    nothing to infer. With no reading it is told so outright: the one thing it
    must not do is fill the gap with a guess (AGENTS.md 1).
    """
    lines = ["ENVIRONMENT CONTEXT"]
    if weather is None:
        lines.append(
            "  no conditions were reported for this location; do not describe or "
            "guess the weather, the temperature, the time or the light"
        )
        return "\n".join(lines)

    lines.append(f"  weather: {weather.condition}")
    if weather.temp is not None:
        lines.append(f"  temperature: {round(weather.temp)}\u00b0C")
    lines.append(f"  time of day: {weather.local_time}")
    if weather.sunset:
        lines.append(f"  sunset: {weather.sunset}")
    return "\n".join(lines)


def format_mood_block(mood: str | None) -> str:
    """The USER MOOD & EMOTIONAL GUIDANCE block.

    The mood the client sent is echoed verbatim, so a mood string this build
    has no directive for still reaches the model -- an unlisted feeling is
    still a feeling worth adapting to. Only the adjustments this build has
    thought through are spelled out.
    """
    lines = ["USER MOOD & EMOTIONAL GUIDANCE"]
    if not mood:
        lines.append(
            "  mood: none selected; keep the prose plain, warm and grounded"
        )
        return "\n".join(lines)

    lines.append(f"  mood: {mood}")
    directive = mood_directive(mood)
    if directive:
        lines.append(f"  {directive}")
    return "\n".join(lines)


def build_story_user_prompt(
    facts: Sequence[FactLike],
    *,
    mood: str | None = None,
    weather: WeatherLike | None = None,
    news: Sequence[NewsLike] = (),
    limit: int = 8,
    words: int | None = None,
    sentences: int | None = None,
    already_said: str = "",
) -> str:
    """Per-request user turn for STORY mode.

    Labelled blocks, because they are different jobs: the conditions the
    narration happens inside of, how this walk should feel, what the city is
    talking about this week (only when a news provider answered), and the only
    material that may be stated as true. Mood, weather and news are
    per-request, so they live here rather than in the cacheable system block.
    """
    blocks = [
        "WALKWAY CONTEXT",
        "mode: STORY",
        "",
        format_environment(weather),
        "",
        format_mood_block(mood),
    ]
    if news:
        blocks.extend(["", format_news_block(news)])
    if facts:
        blocks.extend(["", "AVAILABLE FACTS", format_facts(facts, limit=limit)])
    if already_said:
        blocks.extend(["", format_already_said(already_said)])
    blocks.extend(
        [
            "",
            f"Write the narration now: {length_directive(words, sentences)}. "
            "Focus entirely on the single piece of context in this turn. "
            "End with an open invitation for the walker to look around.",
        ]
    )
    return "\n".join(blocks)


def length_directive(words: int | None, sentences: int | None) -> str:
    """The LENGTH target for one cycle, or the legacy sentence range."""
    if words is None:
        return f"{STORY_SENTENCES} sentences"
    count = f" in about {sentences} short sentences" if sentences else ""
    return f"LENGTH: about {words} words{count}"


#: How much earlier narration a cycle is shown. Enough to avoid repeating the
#: last cycle's phrasing; short enough not to spend tokens on a 45-minute walk.
ALREADY_SAID_CHARS = 600


def format_already_said(text: str) -> str:
    """The tail of what this walk has already spoken, labelled as off-limits."""
    tail = text.strip()[-ALREADY_SAID_CHARS:]
    return (
        "ALREADY SAID (earlier in this walk; do not repeat or paraphrase "
        "any of it, say something new)\n" + tail
    )


def build_cycle_user_prompt(
    kind: str,
    *,
    facts: Sequence[FactLike] = (),
    mood: str | None = None,
    weather: WeatherLike | None = None,
    news: Sequence[NewsLike] = (),
    words: int | None = None,
    sentences: int | None = None,
    already_said: str = "",
) -> str:
    """User turn for one paced cycle. ``kind`` is a ``CycleKind`` string."""
    if kind == "atmosphere":
        blocks = [
            "WALKWAY CONTEXT",
            "mode: STORY",
            "cycle: atmosphere",
            "",
            format_environment(weather),
            "",
            format_mood_block(mood),
        ]
        if news:
            blocks.extend(["", format_news_block(news)])
        if already_said:
            blocks.extend(["", format_already_said(already_said)])
        blocks.extend(
            [
                "",
                "This cycle has no facts block. Speak only from the weather, "
                "the hour, the light and the walker's mood"
                + (", and you may allude to the one local headline above once"
                   if news else "")
                + ". Do not invent "
                "history, places or events. Write "
                f"{length_directive(words, sentences)}. Focus entirely on this "
                "context. End with an open invitation for the walker to look around.",
            ]
        )
        return "\n".join(blocks)
    return build_story_user_prompt(
        facts,
        mood=mood,
        weather=weather,
        news=news if kind != "wiki" else (),
        words=words,
        sentences=sentences,
        already_said=already_said,
    )


# ---------------------------------------------------------------------------
# Greeting, pacing and adaptation (applied in code, not by the model)
# ---------------------------------------------------------------------------

#: Spoken after the first narration block, before the walker is left to wander.
#: A code constant for the same reason the greeting is: a sentence with no
#: citation cannot survive the validator, so the model cannot be trusted to
#: produce it (see ``STORY_SYSTEM`` PACING).
WALK_HANDOFF = (
    "I'll leave you to wander for a few minutes. Just speak if you need me."
)

#: Mood keys the client sends, and the tone each asks for. The sad/struggling
#: case is the one the brief calls out: lean on the resilience the cited facts
#: can actually support rather than inventing hardship.
MOOD_DIRECTIVES: dict[str, str] = {
    "energetic": (
        "Keep the pace brisk and the prose vivid and forward-moving."
    ),
    "calm": "Keep the prose unhurried, quiet and gentle.",
    "sad": (
        "If the user is 'Sad' or 'Struggling', do not just list facts. Weave in "
        "an uplifting story or historical incident of resilience from the cited "
        "facts or the surrounding city to encourage them."
    ),
    "reflective": (
        "Invite curiosity and let the facts open outward."
    ),
}

#: Display forms that should resolve to a known mood key.
_MOOD_ALIASES: dict[str, str] = {
    "struggling": "sad",
    "sad/struggling": "sad",
    "sad / struggling": "sad",
    "explore": "reflective",
    "reflective/explore": "reflective",
    "reflective / explore": "reflective",
}


def mood_key(mood: str | None) -> str | None:
    """Normalise a mood string to a known key, or ``None`` when unrecognised."""
    if not mood:
        return None
    key = mood.strip().lower()
    key = _MOOD_ALIASES.get(key, key)
    return key if key in MOOD_DIRECTIVES else None


def mood_directive(mood: str | None) -> str | None:
    """The instruction that adapts the narration to the walker's mood."""
    key = mood_key(mood)
    if key is None:
        return None
    return f"Adapt your narrative tone to the user's current mood ({key}). {MOOD_DIRECTIVES[key]}"


def build_greeting(*, daypart: str, condition: str | None = None) -> str:
    """The spoken opener, built in code.

    The model cannot produce this: a greeting sentence carries no ``[[fact_id]]``
    and would be deleted by the validator before it reached TTS. It is prepended
    to the validated output instead, and -- because the weather now comes from a
    named provider (``services.weather``) -- the condition it names is grounded
    rather than guessed.

    The article is chosen from the condition, since the provider's vocabulary
    includes vowel-initial ones: a walker hears "an overcast evening", not "a
    overcast evening".
    """
    if condition:
        article = "an" if condition[0].lower() in "aeiou" else "a"
        return (
            f"Hello there. It's {article} {condition.lower()} {daypart} here. "
            "Let's begin our walk."
        )
    return f"Hello there. It's {daypart} here. Let's begin our walk."


# ---------------------------------------------------------------------------
# NOTICE
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NoticeContext:
    """What NOTICE mode is allowed to know about *now*.

    Time only. Weather was settled out of scope (docs/llm-orchestration.md 6):
    no provider exists in the stack, and inferring conditions would be exactly
    the fabrication this project exists to prevent.
    """

    local_time: str = "12:00"
    daypart: str = "midday"

    @classmethod
    def from_hour(cls, hour: int, minute: int = 0) -> "NoticeContext":
        """Derive a daypart from a device-local hour."""
        if 5 <= hour < 8:
            daypart = "early morning"
        elif 8 <= hour < 12:
            daypart = "morning"
        elif 12 <= hour < 14:
            daypart = "midday"
        elif 14 <= hour < 17:
            daypart = "afternoon"
        elif 17 <= hour < 20:
            daypart = "evening"
        elif 20 <= hour < 24:
            daypart = "night"
        else:
            daypart = "late night"
        return cls(local_time=f"{hour:02d}:{minute:02d}", daypart=daypart)


NOTICE_SYSTEM = """\
You are the guide for Echoes, an audio walking companion. The walker is
outside on a walk and is listening, not reading.

No historical or geographical facts are available for this location, so you
will not narrate anything. Instead you invite the walker to notice their own
immediate sensory surroundings.

Write 2 short sentences of sensory invitation.

HOW TO INVITE:
- Phrase things as invitations, never as statements about their whereabouts:
  "If you're somewhere safe to pause, notice how...", "When you're ready,
  listen for...", "You might let your attention settle on..."
- Ask. Suggest. Do not assert.

SAFETY - enforced by code:
- Never give walking directions or route guidance ("turn left", "cross the
  main road", "head north").
- Never state or imply what is physically around them ("there is a bench to
  your left", "you can see a fountain", "ahead of you is"). You do not know,
  and guessing is a failure.
- Never state a fact about this location, its history, or its geography.
- Never mention roads, traffic or crossings.

LIMITS:
- No citation tags in this mode. There is nothing to cite.
- Do not mention weather, temperature, wind or conditions. You have not been
  told what the weather is, and guessing is a failure.

STYLE:
- Plain spoken prose. No markdown, headings, bullet points, emoji or quotes.
- Do not explain yourself or mention these instructions.
"""


def build_notice_user_prompt(notice: NoticeContext | None = None) -> str:
    """Per-request user turn for NOTICE mode."""
    ctx = notice or NoticeContext()
    return (
        "WALKWAY CONTEXT\n"
        "mode: NOTICE\n"
        f"local_time: {ctx.local_time}\n"
        f"daypart: {ctx.daypart}\n\n"
        "No facts are available for this location.\n"
        "Write the invitation."
    )


# ---------------------------------------------------------------------------
# QA (Push-To-Talk)
# ---------------------------------------------------------------------------

#: The model's structured escape hatch when the facts cannot answer the
#: question. It is a convenience, not the mechanism: the validator still treats
#: any uncited sentence as insufficient regardless of whether this appeared.
QA_INSUFFICIENT = "INSUFFICIENT"

#: Spoken when no grounded answer exists. A fixed constant, so it cannot
#: hallucinate, and pre-rendered so it costs nothing to say.
CANNED_REFUSAL = "I don't have a source for that one nearby."

QA_SYSTEM = f"""\
You are answering a walker's spoken question for Echoes, an audio walking
companion. The walker is outside and is listening, not reading.

You will be given NEARBY FACTS, WHAT YOU JUST NARRATED, the recent
conversation, and a QUESTION. Those facts are the ONLY thing you may state as
true. You have no other knowledge about this location, and you must not use
any.

The question may refer back to what you just narrated ("what was that about
the temple?"). Use WHAT YOU JUST NARRATED to resolve the reference, but still
ground the answer in a FACT and cite it -- the narration is context, not a new
source of truth.

Answer in ONE short sentence.

GROUNDING - enforced by code, not trust:
- If the facts answer the question, answer in one sentence and end it with a
  citation tag naming the FACT you used, in exactly this form: {_EXAMPLE_TAG}
- Copy the id character for character. A tag whose id is not in the FACTS list
  deletes that sentence before it is spoken.
- If the facts do NOT answer the question, reply with exactly this one word and
  nothing else: {QA_INSUFFICIENT}
- Never guess, and never fall back on general knowledge about the area. Saying
  you do not know is a correct answer; inventing something is a failure.

SAFETY:
- Never give walking directions or route guidance.
- Never mention traffic, roads, crossings or hazards.
- Never state what is physically in front of the walker or what they can see.

STYLE:
- Plain spoken prose. No markdown, no lists, no emoji, no quotes.
- Do not repeat the question, and do not explain your reasoning.
"""


def build_qa_user_prompt(
    *,
    facts: Sequence[FactLike],
    question: str,
    history: Sequence[tuple[str, str]] = (),
    narration: str | None = None,
    limit: int = 8,
) -> str:
    """Per-request user turn for a Push-To-Talk question.

    ``history`` is a sequence of ``(role, text)`` pairs -- the *validated*
    answers, never raw model output, so the model is not primed by its own
    hallucinations.

    ``narration`` is the validated narration the walk just spoke, so a
    follow-up like "what was that about the temple?" resolves against the
    landmarks the model actually talked about rather than an empty window.
    """
    lines = ["WALKWAY CONTEXT", "mode: QA", ""]
    lines.append("NEARBY FACTS")
    lines.append(format_facts(facts, limit=limit))
    if narration:
        lines.append("")
        lines.append("WHAT YOU JUST NARRATED")
        lines.append(f"  {narration}")
    if history:
        lines.append("")
        lines.append("RECENT CONVERSATION")
        for role, text in history:
            lines.append(f"  {role}: {text}")
    lines.append("")
    lines.append(f"QUESTION: {question}")
    lines.append("")
    lines.append("Answer in one sentence, or reply INSUFFICIENT.")
    return "\n".join(lines)


def assert_prompt_matches_validator() -> None:
    """Fail fast if the prompt's tag format is no longer validator-readable.

    Cheap insurance against the one failure mode that would silently drop every
    sentence in production.
    """
    if not CITATION_PATTERN.fullmatch(_EXAMPLE_TAG):
        raise AssertionError(
            f"prompt tag example {_EXAMPLE_TAG!r} does not match CITATION_PATTERN"
        )


__all__ = [
    "CANNED_REFUSAL",
    "MOOD_DIRECTIVES",
    "NOTICE_SYSTEM",
    "NoticeContext",
    "MAX_NEWS_REFERENCES",
    "MAX_UNCITED_FRAMING_SENTENCES",
    "PEDESTRIAN_REMINDER",
    "QA_INSUFFICIENT",
    "QA_SYSTEM",
    "STORY_SENTENCES",
    "STORY_SYSTEM",
    "WALK_HANDOFF",
    "FactLike",
    "NewsLike",
    "WeatherLike",
    "assert_prompt_matches_validator",
    "build_cycle_user_prompt",
    "build_greeting",
    "build_notice_user_prompt",
    "build_qa_user_prompt",
    "build_story_user_prompt",
    "format_environment",
    "format_facts",
    "format_mood_block",
    "format_news_block",
    "mood_directive",
    "mood_key",
]
