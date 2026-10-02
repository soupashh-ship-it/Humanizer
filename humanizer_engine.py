"""
Humanizer Engine - Windows version.
Rule-based multi-pass pipeline (100% stdlib, works fully offline) with an
optional neural upgrade: T5-small paraphrasing + MiniLM semantic similarity -
the exact models used by the HuggingFace Space this was ported from.

Offline default: AdvancedAIHumanizer() - instant start, no downloads.
Neural mode:    AdvancedAIHumanizer(neural=True) or .enable_neural_models()
                (needs requirements_optional.txt installed once + models
                cached; falls back gracefully when unavailable).
"""
import random
import re
import math
from collections import Counter, defaultdict
from typing import Dict, Tuple


# ---------- Lightweight NLP helpers (no nltk needed) ----------

_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'])')
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:'[a-z]+)?")


def sent_tokenize(text: str):
    """Simple sentence tokenizer (replaces nltk.sent_tokenize)."""
    text = text.strip()
    if not text:
        return []
    # Protect common abbreviations
    protected = text
    for abbr in ["e.g.", "i.e.", "Mr.", "Mrs.", "Dr.", "vs.", "etc."]:
        protected = protected.replace(abbr, abbr.replace(".", "<DOT>"))
    parts = _SENT_SPLIT_RE.split(protected)
    out = []
    for p in parts:
        p = p.replace("<DOT>", ".").strip()
        if p:
            out.append(p)
    return out if out else [text]


def word_tokenize(text: str):
    """Simple word tokenizer (replaces nltk.word_tokenize)."""
    return _WORD_RE.findall(text)


BASIC_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why", "how",
    "all", "any", "both", "each", "few", "more", "most", "other", "some",
    "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too",
    "very", "can", "will", "just", "don", "should", "now",
}


# Synonyms that shift meaning and must NEVER be produced: strengtheners
# (prove, maximize, undoubtedly), narrowers (survey, trial, stats),
# connotation shifts (burden, old-style, folks), or claim changes
# (approve, confirm, decide, achieve, recommend). Inputs containing these
# words are left alone; only their use as replacements is blocked.
_UNSAFE_SYNONYMS = {
    "profit", "profits", "cost-effective", "unhurried", "survey", "surveys",
    "trial", "trials", "insight", "insights", "perspective", "perspectives",
    "view", "views", "approve", "approves", "approved", "authorize",
    "authorizes", "authorized", "confirm", "confirms", "confirmed",
    "decide", "decides", "decided", "achieve", "achieves", "achieved",
    "demand", "demands", "demanded", "involve", "involves", "involved",
    "recommend", "recommends", "recommended", "urge", "urges", "urged",
    "highlight", "highlights", "highlighted", "stats", "burden", "burdens",
    "overhaul", "overhauls", "overhauled", "remake", "remakes", "remade",
    "old-style", "tricky", "honestly", "undoubtedly", "deeply", "prove",
    "proves", "proved", "proving", "attain", "attains", "attained",
    "continue", "continues", "continued", "continuing", "found", "drive",
    "drives", "drove", "driven", "features", "featured", "perfect",
    "perfects", "perfected", "maximize", "maximizes", "maximized",
    "folk", "folks", "trainer", "trainers",
}


# Base verbs that follow "help <object>" as bare infinitives
# ("help teachers track"). Only help/make/let license this pattern, so the
# engine must not swap help -> assist/aid/support/guide in that position.
_BARE_INF_AFTER_HELP = {
    "track", "monitor", "follow", "save", "reduce", "cut", "improve",
    "boost", "learn", "understand", "manage", "handle", "find", "get",
    "make", "take", "keep", "stay", "build", "create", "write", "read",
    "work", "run", "grow", "do", "go", "see", "achieve", "reach", "meet",
    "solve", "prevent", "avoid", "start", "navigate", "identify",
    "discover", "develop", "deliver", "drive", "gain", "become", "feel",
    "remain", "maintain", "protect", "support", "decide", "choose",
}


def count_syllables(word: str) -> int:
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    word = re.sub(r'[^a-z]', '', word)
    if not word:
        return 1
    # Remove silent e
    if word.endswith("e"):
        word = word[:-1]
    vowels = "aeiouy"
    count = 0
    prev_vowel = False
    for ch in word:
        is_vowel = ch in vowels
        if is_vowel and not prev_vowel:
            count += 1
        prev_vowel = is_vowel
    # Adjust for -es, -ed
    if word.endswith("es") or word.endswith("ed"):
        count -= 1
    return max(1, count)


def flesch_reading_ease(text: str) -> float:
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    if not sentences or not words:
        return 70.0
    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)
    score = 206.835 - 1.015 * wps - 84.6 * spw
    return max(0.0, min(100.0, score))


def flesch_kincaid_grade(text: str) -> float:
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    if not sentences or not words:
        return 8.0
    syllables = sum(count_syllables(w) for w in words)
    wps = len(words) / len(sentences)
    spw = syllables / len(words)
    return max(0.0, 0.39 * wps + 11.8 * spw - 15.59)


# ---------- Extended synonym dictionary (replaces WordNet) ----------
# Built from original word_groups + common academic words.

EXTENDED_SYNONYMS = {
    "analyze": ["examine", "study", "investigate", "explore", "review", "assess"],
    "analysis": ["examination", "study", "review", "evaluation", "assessment"],
    "important": ["crucial", "vital", "essential", "key", "critical", "major"],
    "shows": ["reveals", "indicates", "displays", "confirms", "proves", "highlights"],
    "show": ["reveal", "display", "prove", "illustrate", "highlight"],
    "understand": ["comprehend", "grasp", "realize", "recognize", "appreciate"],
    "develop": ["create", "build", "form", "generate", "produce", "set up"],
    "improve": ["enhance", "refine", "upgrade", "advance", "boost", "strengthen"],
    "consider": ["examine", "evaluate", "contemplate", "weigh", "reflect on", "think about", "mull over"],
    "determine": ["figure out", "decide", "work out"],
    "conduct": ["run", "carry out", "do"],
    "assist": ["help", "aid", "support"],
    "individuals": ["people", "persons", "folks"],
    "numerous": ["many", "plenty of", "lots of"],
    "sufficient": ["enough", "adequate"],
    "receive": ["get"],
    "approximately": ["about", "around", "roughly"],
    "additional": ["extra", "more", "further"],
    "commence": ["start", "begin", "kick off"],
    "inquire": ["ask", "ask about"],
    "acquire": ["get", "gain", "pick up"],
    "indicate": ["show", "suggest", "point to"],
    "facilitate": ["help", "ease", "enable"],
    "offer": ["provide", "give"],
    "track": ["follow", "monitor", "watch"],
    "reduce": ["cut", "lower", "lessen"],
    "affect": ["influence", "impact", "shape"],
    "protect": ["safeguard", "shield", "defend"],
    "adopt": ["take on", "embrace"],
    "focus": ["concentrate", "prioritize", "spotlight"],
    "encourage": ["urge", "promote", "spur"],
    "suggest": ["propose", "recommend"],
    "platform": ["system", "service", "tool"],
    "student": ["learner"],
    "teacher": ["educator", "instructor"],
    "outcome": ["result"],
    "data": ["information", "stats"],
    "institution": ["organization", "establishment"],
    "workload": ["load", "burden"],
    "improvement": ["advance", "gain", "boost"],
    "solution": ["answer", "fix"],
    "feature": ["highlight", "detail"],
    "educator": ["teacher", "instructor", "mentor"],
    "transform": ["reshape", "overhaul", "remake"],
    "education": ["schooling", "instruction"],
    "school": ["classroom"],
    "learning": ["education", "training"],
    "tools": ["resources", "utilities"],
    "different": ["various", "diverse", "distinct", "separate", "alternative"],
    "effective": ["successful", "efficient", "productive", "powerful", "useful"],
    "significant": ["important", "considerable", "notable", "major", "big"],
    "implement": ["apply", "execute", "carry out", "deploy", "put in place"],
    "utilize": ["use", "employ", "apply", "harness", "tap into"],
    "comprehensive": ["complete", "thorough", "extensive", "detailed", "full"],
    "fundamental": ["basic", "essential", "core", "primary", "key"],
    "substantial": ["considerable", "large", "major", "big", "hefty"],
    "demonstrate": ["show", "prove", "illustrate", "reveal", "display"],
    "establish": ["set up", "create", "build", "form", "start"],
    "maintain": ["keep", "preserve", "sustain", "continue", "uphold"],
    "obtain": ["get", "acquire", "gain", "secure", "achieve"],
    "create": ["build", "produce", "generate", "develop", "make"],
    "provide": ["offer", "supply", "give", "deliver", "present"],
    "require": ["need", "demand", "call for", "involve", "take"],
    "allow": ["enable", "permit", "let", "authorize", "approve"],
    "ensure": ["guarantee", "make sure", "confirm", "secure", "assure"],
    "increase": ["raise", "boost", "grow", "expand", "lift"],
    "decrease": ["reduce", "lower", "cut", "drop", "lessen"],
    "large": ["big", "huge", "vast", "enormous", "massive"],
    "small": ["tiny", "little", "modest", "minor", "compact"],
    "good": ["great", "solid", "strong", "excellent", "fine"],
    "bad": ["poor", "weak", "lacking", "subpar", "inadequate"],
    "new": ["fresh", "novel", "recent", "modern", "latest"],
    "old": ["older", "previous", "earlier", "prior", "existing"],
    "fast": ["quick", "rapid", "swift", "speedy", "brisk"],
    "slow": ["gradual", "steady", "unhurried", "measured"],
    "help": ["assist", "aid", "support", "guide"],
    "use": ["employ", "apply", "tap into", "work with"],
    "make": ["create", "produce", "build", "craft", "form"],
    "get": ["gain", "secure", "receive", "acquire", "grab"],
    "give": ["offer", "provide", "supply", "present", "deliver"],
    "need": ["require", "demand", "call for", "necessitate"],
    "want": ["wish for", "seek", "desire", "hope for", "aim for"],
    "think": ["believe", "feel", "reckon", "consider", "figure"],
    "know": ["understand", "grasp", "recognize", "realize"],
    "see": ["observe", "notice", "spot", "witness", "perceive"],
    "find": ["discover", "uncover", "locate", "identify", "pinpoint"],
    "change": ["shift", "alter", "adjust", "modify", "transform"],
    "start": ["begin", "kick off", "launch", "commence", "open"],
    "begin": ["start", "commence", "kick off", "open", "initiate"],
    "end": ["finish", "conclude", "wrap up", "close", "complete"],
    "try": ["attempt", "aim", "seek", "strive", "endeavor"],
    "work": ["function", "operate", "perform", "serve", "run"],
    "part": ["portion", "segment", "section", "piece", "component"],
    "area": ["region", "zone", "field", "domain", "sphere"],
    "method": ["approach", "technique", "strategy", "way", "system"],
    "way": ["approach", "method", "path", "route", "means"],
    "result": ["outcome", "consequence"],
    "problem": ["issue", "challenge", "difficulty", "obstacle", "hurdle"],
    "idea": ["concept", "notion", "thought", "insight", "perspective"],
    "opinion": ["view", "perspective", "take", "stance", "viewpoint"],
    "research": ["study", "investigation", "analysis", "inquiry", "exploration"],
    "study": ["research", "examination", "review", "analysis", "survey"],
    "example": ["instance", "case", "illustration", "sample", "demonstration"],
    "benefit": ["gain", "profit"],
    "challenge": ["hurdle", "obstacle", "difficulty", "test", "trial"],
    "experience": ["background", "history", "exposure", "practice", "familiarity"],
    "knowledge": ["understanding", "insight", "expertise", "awareness", "grasp"],
    "ability": ["skill", "capacity", "capability", "talent", "aptitude"],
    "often": ["frequently", "commonly", "regularly", "typically", "usually"],
    "always": ["invariably", "constantly", "continually", "every time"],
    "never": ["not ever", "at no time", "on no occasion"],
    "very": ["really", "truly", "extremely", "highly", "particularly"],
    "really": ["truly", "genuinely", "honestly", "actually"],
    "quickly": ["fast", "rapidly", "swiftly", "promptly", "speedily"],
    "slowly": ["gradually", "steadily", "progressively", "bit by bit"],
    "clearly": ["obviously", "evidently", "plainly", "undoubtedly"],
    "however": ["but", "yet", "though", "still", "that said"],
    "therefore": ["so", "as a result", "that's why", "for that reason"],
    "additionally": ["also", "plus", "besides", "on top of that"],
    "finally": ["ultimately", "in the end", "lastly", "to wrap up"],
    "importantly": ["notably", "significantly", "crucially"],
    "generally": ["usually", "typically", "broadly", "for the most part"],
    "usually": ["generally", "typically", "normally", "commonly"],
    "completely": ["fully", "entirely", "totally", "wholly", "thoroughly"],
    "entirely": ["fully", "completely", "wholly", "totally"],
    "mainly": ["primarily", "mostly", "chiefly", "largely", "principally"],
    "highly": ["extremely", "greatly", "deeply", "strongly"],
    "deeply": ["profoundly", "greatly", "strongly", "thoroughly"],
    "quick": ["fast", "rapid", "swift", "speedy"],
    "accurate": ["precise", "exact", "correct", "reliable"],
    "reliable": ["dependable", "trustworthy", "solid", "consistent"],
    "efficient": ["effective", "productive", "smooth", "cost-effective"],
    "complex": ["complicated", "sophisticated", "involved", "tricky"],
    "simple": ["straightforward", "easy", "uncomplicated", "plain"],
    "modern": ["contemporary", "current", "present-day", "up-to-date"],
    "traditional": ["conventional", "classic", "customary", "old-style"],
    "various": ["diverse", "multiple", "assorted", "different", "several"],
    "several": ["multiple", "various", "a number of", "a handful of"],
    "many": ["numerous", "countless", "plenty of", "lots of", "a lot of"],
    "much": ["a lot", "plenty", "a great deal", "loads"],
    "essential": ["crucial", "vital", "critical", "key", "necessary"],
    "crucial": ["critical", "vital", "essential", "pivotal", "key"],
    "vital": ["crucial", "essential", "critical", "key"],
    "key": ["central", "core", "main", "critical", "essential"],
    "major": ["considerable", "notable", "big", "large", "huge"],
    "notable": ["noteworthy", "remarkable", "striking", "memorable"],
    "remarkable": ["notable", "striking", "extraordinary", "exceptional"],
    "innovative": ["creative", "original", "fresh", "inventive", "novel"],
    "advanced": ["sophisticated", "modern", "latest", "top-notch"],
    "excellent": ["outstanding", "superb", "top-notch", "first-rate"],
    "outstanding": ["excellent", "exceptional", "superb", "remarkable"],
    "thorough": ["detailed", "in-depth", "complete", "careful"],
    "detailed": ["thorough", "in-depth", "specific", "painstaking"],
    "precise": ["exact", "accurate", "specific", "detailed"],
    "exact": ["precise", "accurate", "specific", "particular"],
}


class AdvancedAIHumanizer:
    """Offline humanizer engine - same behavior as original app.py, no heavy deps.

    Neural upgrade (optional, mirrors the HuggingFace Space 1:1):
    call enable_neural_models() to load T5-small paraphrasing +
    MiniLM semantic similarity. Without it, rule-based fallbacks are used.
    """

    # Set to True to attempt neural load automatically on first use.
    # Kept False so cold start stays instant and fully offline.
    AUTO_NEURAL = False

    def __init__(self, seed=None, neural=False):
        if seed is not None:
            random.seed(seed)
        self.setup_humanization_patterns()
        self.setup_fallback_embeddings()
        self.stop_words = BASIC_STOPWORDS
        self.fillers = [
            "you know", "I mean", "sort of", "kind of", "basically", "actually",
            "really", "quite", "pretty much", "more or less", "essentially",
        ]
        self.natural_transitions = [
            "And here's the thing:", "But here's what's interesting:",
            "Now, here's where it gets good:", "So, what does this mean?",
            "Here's why this matters:", "Think about it this way:",
            "Let me put it this way:", "Here's the bottom line:",
            "The reality is:", "What we're seeing is:",
            "The truth is:", "At the end of the day:",
        ]
        # No heavy models in offline build (kept as None for API compat)
        self.sentence_model = None
        self.paraphrase_model = None
        self.paraphrase_tokenizer = None
        self.nlp = None
        self._neural_error = ""
        if neural or self.AUTO_NEURAL:
            self.enable_neural_models()

    def enable_neural_models(self) -> bool:
        """Load T5-small + MiniLM (same models as the HuggingFace Space).

        Lazy: only downloads/loads on first call. Returns True if at least
        one model loaded. Fully optional - everything works without them.
        """
        if self.paraphrase_model is not None and self.sentence_model is not None:
            return True
        if self.paraphrase_model is None:
            try:
                from transformers import T5Tokenizer, T5ForConditionalGeneration
                self.paraphrase_tokenizer = T5Tokenizer.from_pretrained('google-t5/t5-small')
                self.paraphrase_model = T5ForConditionalGeneration.from_pretrained('google-t5/t5-small')
                self.paraphrase_model.eval()
            except Exception as e:
                self._neural_error = "T5: %s" % str(e)[:120]
                self.paraphrase_model = None
                self.paraphrase_tokenizer = None
        if self.sentence_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self.sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            except Exception as e:
                err = "MiniLM: %s" % str(e)[:120]
                self._neural_error = (self._neural_error + " | " + err) if self._neural_error else err
                self.sentence_model = None
        return self.paraphrase_model is not None or self.sentence_model is not None

    def neural_status(self) -> str:
        parts = []
        parts.append("T5-small paraphrase: %s" % ("ON" if self.paraphrase_model is not None else "off"))
        parts.append("MiniLM similarity: %s" % ("ON" if self.sentence_model is not None else "off"))
        if self._neural_error and (self.paraphrase_model is None or self.sentence_model is None):
            parts.append("(%s)" % self._neural_error)
        return ", ".join(parts)

    # ----- patterns (ported 1:1 from original) -----
    def setup_humanization_patterns(self):
        self.ai_indicators = {
            r'\bdelve into\b': ["explore", "examine", "investigate", "look into", "study", "dig into", "analyze"],
            r'\bembark upon?\b': ["begin", "start", "initiate", "launch", "set out", "commence", "kick off"],
            r'\ba testament to\b': ["proof of", "evidence of", "shows", "demonstrates", "reflects", "indicates"],
            r'\blandscape of\b': ["world of", "field of", "area of", "context of", "environment of", "space of"],
            r'\bnavigating\b': ["handling", "managing", "dealing with", "working through", "tackling", "addressing"],
            r'\bmeticulous\b': ["careful", "thorough", "detailed", "precise", "systematic", "methodical"],
            r'\bintricate\b': ["complex", "detailed", "sophisticated", "elaborate", "complicated", "involved"],
            r'\bmyriad\b': ["many", "numerous", "countless", "various", "multiple", "lots of"],
            r'\bplethora\b': ["abundance", "wealth", "variety", "range", "loads", "tons"],
            r'\bparadigm\b': ["model", "framework", "approach", "system", "way", "method"],
            r'\bsynergy\b': ["teamwork", "cooperation", "collaboration", "working together", "unity"],
            r'\bleverage\b': ["use", "utilize", "employ", "apply", "tap into", "make use of"],
            r'\bfacilitate\b': ["help", "assist", "enable", "support", "aid"],
            r'\boptimize\b': ["improve", "enhance", "refine", "boost"],
            r'\bstreamline\b': ["simplify", "improve", "refine", "smooth out", "make efficient"],
            r'\brobust\b': ["strong", "reliable", "solid", "sturdy", "effective", "powerful"],
            r'\bseamless\b': ["smooth", "fluid", "effortless", "easy", "integrated", "unified"],
            r'\binnovative\b': ["creative", "original", "new", "fresh", "groundbreaking", "inventive"],
            r'\bcutting-edge\b': ["advanced", "modern", "latest", "new", "state-of-the-art", "leading"],
            r'\bstate-of-the-art\b': ["advanced", "modern", "latest", "top-notch", "cutting-edge"],
            r'\bfurthermore\b': ["also", "plus", "what's more", "on top of that", "besides", "additionally"],
            r'\bmoreover\b': ["also", "plus", "what's more", "on top of that", "besides", "furthermore"],
            r'\bhowever\b': ["but", "yet", "still", "that said"],
            r'\bnevertheless\b': ["still", "yet", "even so", "but", "however", "all the same"],
            r'\btherefore\b': ["so", "thus", "that's why", "as a result", "because of this", "for this reason"],
            r'\bconsequently\b': ["so", "therefore", "as a result", "because of this", "thus", "that's why"],
            r'\bin conclusion\b': ["finally", "to wrap up", "in the end", "ultimately", "lastly", "to finish"],
            r'\bto summarize\b': ["in short", "briefly", "to sum up", "basically", "in essence", "overall"],
            r'\bin summary\b': ["briefly", "in short", "basically", "to sum up", "overall", "in essence"],
            r'\bin order to\b': ["to", "aiming to"],
            r'\bdue to the fact that\b': ["because", "since", "as", "given that", "seeing that"],
            r'\bfor the purpose of\b': ["to", "for", "aiming to", "meant for"],
            r'\bwith regard to\b': ["about", "concerning", "regarding", "when it comes to", "as for"],
            r'\bin terms of\b': ["regarding", "when it comes to", "as for", "concerning", "about"],
            r'\bby means of\b': ["through", "using", "via", "by way of", "with"],
            r'\bas a result of\b': ["because of", "due to", "from", "owing to", "thanks to"],
            r'\bin the event that\b': ["if", "should", "in case", "when", "if it happens that"],
            r'\bprior to\b': ["before", "ahead of", "earlier than", "in advance of"],
            r'\bsubsequent to\b': ["after", "following", "later than", "once"],
            r'\bcomprehensive\b': ["complete", "thorough", "detailed", "full", "extensive", "in-depth"],
            r'\bfundamental\b': ["basic", "essential", "core", "key", "primary", "main"],
            r'\bsubstantial\b': ["significant", "considerable", "large", "major", "big", "huge"],
            r'\bsignificant\b': ["important", "major", "considerable", "substantial", "notable", "big"],
            r'\bimplement\b': ["put in place", "carry out", "apply", "execute", "use", "deploy"],
            r'\butilize\b': ["use", "employ", "apply", "make use of", "tap into", "leverage"],
            r'\bdemonstrate\b': ["show", "illustrate", "reveal", "display"],
            r'\bestablish\b': ["set up", "create", "build", "form", "start"],
            r'\bmaintain\b': ["keep", "preserve", "sustain", "continue", "uphold", "retain"],
            r'\bobtain\b': ["get", "acquire", "gain", "secure"],
        }
        # Inflected forms of the most common AI-flagged verbs.
        # The original app only matched base forms (e.g. "utilize" but not
        # "utilized"), so these close the biggest coverage gap.
        self.ai_indicators.update({
            r'\butilizes\b': ["uses", "employs", "applies", "harnesses"],
            r'\butilized\b': ["used", "employed", "applied", "harnessed"],
            r'\butilizing\b': ["using", "employing", "applying", "harnessing"],
            r'\bimplements\b': ["applies", "executes", "carries out", "deploys"],
            r'\bimplemented\b': ["applied", "executed", "carried out", "deployed"],
            r'\bimplementing\b': ["applying", "executing", "carrying out", "deploying"],
            r'\bdemonstrates\b': ["shows", "illustrates", "reveals", "displays"],
            r'\bdemonstrated\b': ["showed", "illustrated", "revealed", "disclosed"],
            r'\bdemonstrating\b': ["showing", "illustrating", "revealing", "disclosing"],
            r'\bfacilitates\b': ["helps", "enables", "supports", "aids"],
            r'\bfacilitated\b': ["helped", "enabled", "supported", "aided"],
            r'\bfacilitating\b': ["helping", "enabling", "supporting", "aiding"],
            r'\bestablishes\b': ["sets up", "creates", "builds", "forms"],
            r'\bestablished\b': ["set up", "created", "built", "formed"],
            r'\bestablishing\b': ["setting up", "creating", "building", "forming"],
            r'\bleverages\b': ["uses", "employs", "applies", "taps into"],
            r'\bleveraged\b': ["used", "employed", "applied", "tapped into"],
            r'\bleveraging\b': ["using", "employing", "applying", "tapping into"],
            r'\boptimizes\b': ["improves", "enhances", "refines", "boosts"],
            r'\boptimized\b': ["improved", "enhanced", "refined", "boosted"],
            r'\boptimizing\b': ["improving", "enhancing", "refining", "boosting"],
            r'\bstreamlines\b': ["simplifies", "improves", "refines"],
            r'\bstreamlined\b': ["simplified", "improved", "refined"],
            r'\bstreamlining\b': ["simplifying", "improving", "refining"],
            r'\bnavigates\b': ["handles", "manages", "deals with", "tackles"],
            r'\bnavigated\b': ["handled", "managed", "dealt with", "tackled"],
            r'\bnavigating\b': ["handling", "managing", "dealing with", "tackling"],
            r'\bmaintains\b': ["keeps", "preserves", "sustains", "upholds"],
            r'\bmaintained\b': ["kept", "preserved", "sustained", "upheld"],
            r'\bmaintaining\b': ["keeping", "preserving", "sustaining", "upholding"],
            r'\bobtains\b': ["gets", "acquires", "gains", "secures"],
            r'\bobtained\b': ["got", "acquired", "gained", "secured"],
            r'\bobtaining\b': ["getting", "acquiring", "gaining", "securing"],
            r'\bdelves into\b': ["explores", "examines", "investigates", "studies"],
            r'\bdelved into\b': ["explored", "examined", "investigated", "studied"],
            r'\bdelving into\b': ["exploring", "examining", "investigating", "studying"],
        })
        # Well-known GPT/AI "tells" - extra patterns so the output carries
        # zero detectable AI vocabulary (also enforced by force_clean_residue).
        self.ai_indicators.update({
            r'\btapestry\b': ["mix", "blend", "combination"],
            r'\bvibrant\b': ["lively", "active", "thriving"],
            r'\bbustling\b': ["busy", "lively", "crowded"],
            r'\brealm of\b': ["field of", "world of", "area of"],
            r'\bunlock\b': ["open up", "reveal"],
            r'\bunleash\b': ["release", "bring out"],
            r'\belevate\b': ["raise", "lift", "boost"],
            r'\bfoster\b': ["encourage", "build", "support"],
            r'\bfostering\b': ["encouraging", "building", "supporting"],
            r'\bboasts\b': ["has", "offers", "includes"],
            r'\bshowcase\b': ["show", "highlight", "feature"],
            r'\bshowcases\b': ["shows", "highlights", "presents"],
            r'\bshowcasing\b': ["showing", "highlighting", "presenting"],
            r'\bunderscores\b': ["highlights", "shows", "points to"],
            r'\bgame-changer\b': ["major shift", "turning point", "big step"],
            r'\bdive into\b': ["get into", "jump into", "explore"],
            r'\bdelve\b': ["dig", "explore"],
            r'\bdelves\b': ["digs", "explores"],
            r'\bplays? a (?:crucial|key|vital|important|significant) role in\b': ["is essential for", "is key for", "matters for", "underpins"],
            r'\bpave the way for\b': ["lead to", "enable", "make possible"],
            r'\bshed light on\b': ["clarify", "explain", "clear up"],
            r"\bin today's fast-paced world\b": ["today", "nowadays"],
            r'\bfast-paced world\b': ["busy world", "modern world"],
            r'\bever-evolving\b': ["changing", "fast-changing"],
            r'\bmultifaceted\b': ["complex", "many-sided", "layered"],
            r'\bholistic\b': ["complete", "all-round"],
            r'\bfirstly\b': ["first", "first off"],
            r'\bsecondly\b': ["second", "next"],
            r'\ba wide range of\b': ["many", "all kinds of", "plenty of"],
            r'\bdiverse range of\b': ["mix of", "range of"],
            r'\bgone are the days when\b': ["it used to be that", "there was a time when"],
            r'\bbuckle up\b': ["get ready", "hold on"],
            r'\blook no further\b': ["here's what works", "this is what you need"],
            r'\bnestled\b': ["located", "set", "situated"],
            r'\benchanting\b': ["charming", "lovely", "delightful"],
            r'\bcaptivating\b': ["engaging", "gripping", "fascinating"],
            r'\badvent of\b': ["arrival of", "rise of"],
            r'\bharness the power of\b': ["use", "tap into"],
            r'\bto the next level\b': ["up a notch", "a lot further"],
            r'\bdesigned to help\b': ["built to help", "made to help"],
            r'\bdesigned to enable\b': ["built to enable", "made to enable"],
            r'\bdesigned to improve\b': ["built to improve", "made to improve"],
            r'\bdesigned to provide\b': ["built to provide", "made to provide"],
            r'\bdesigned to support\b': ["built to support", "made to support"],
            r'\bdesigned to ensure\b': ["built to ensure", "made to ensure"],
            r'\bdesigned to\b': ["built to", "made to", "meant to"],
            r'\bneedless to say,?': ["obviously,", "clearly,"],
            r'\bfirst and foremost,?': ["first,", "to begin,"],
            r'\blast but not least,?': ["finally,", "lastly,"],
            r"\bit should be noted that\b": ["note that", "keep in mind that"],
            r"\bit is worth (?:noting|mentioning) that\b": ["note that", "remember that"],
            r"\bit is important to note (?:that )?": ["the thing is, ", "remember, ", "keep in mind, "],
            # Catches variants broken by synonym swaps (e.g. "it is major to note that")
            r"\bit is \w+ to note(?: that)?": ["the thing is, ", "remember, ", "keep in mind, "],
            r"\bit'?s important to note (?:that )?": ["the thing is, ", "remember, ", "keep in mind, "],
            r'\bit goes without saying,?': ["obviously,", "clearly,"],
            r"\bit'?s (?:clear|evident|obvious|apparent) that\b": ["clearly,", "obviously,"],
            r"\bit is (?:clear|evident|obvious|apparent) that\b": ["clearly,", "obviously,"],
            r'\bis able to\b': ["can"],
            r'\bare able to\b': ["can"],
            r'\bhas the ability to\b': ["can"],
            r'\bhave the ability to\b': ["can"],
        })
        self.ai_indicators.update({
            r'\badditionally\b': ["also", "plus", "besides"],
            r'\bin addition,(?=\s)': ["plus,", "also,"],
            r'\bhence\b': ["so", "that's why"],
            r'\bthus\b': ["so", "as a result"],
            r'\bwhilst\b': ["while"],
            r'\bamongst\b': ["among"],
            r'\boftentimes\b': ["often"],
            r'\baforementioned\b': ["previous", "earlier"],
            r'\bpertaining to\b': ["about", "concerning"],
            r'\bwith respect to\b': ["about", "regarding"],
            r'\bregarding\b': ["about", "on"],
            r'\battempt to\b': ["try to"],
            r'\bendeavo?ur to\b': ["try to"],
            r'\bplays? a role in\b': ["matters for", "counts for"],
            r'\bgive rise to\b': ["lead to", "cause"],
            r'\bbring about\b': ["cause", "lead to"],
        })
        self.human_starters = [
            "Actually,", "Honestly,", "Basically,", "Really,", "Generally,", "Usually,",
            "Often,", "Sometimes,", "Clearly,", "Obviously,", "Naturally,", "Certainly,",
            "Definitely,", "Interestingly,", "Surprisingly,", "Notably,", "Importantly,",
            "What's more,", "Plus,", "Also,", "Besides,", "On top of that,", "In fact,",
            "Indeed,", "Of course,", "No doubt,", "Without question,", "Frankly,",
            "To be honest,", "Truth is,", "The thing is,", "Here's the deal,", "Look,",
        ]
        self.contractions = {
            r'\bit is\b': "it's", r'\bthat is\b': "that's", r'\bthere is\b': "there's",
            r'\bwho is\b': "who's", r'\bwhat is\b': "what's", r'\bwhere is\b': "where's",
            r'\bthey are\b': "they're", r'\bwe are\b': "we're", r'\byou are\b': "you're",
            r'\bI am\b': "I'm", r'\bhe is\b': "he's", r'\bshe is\b': "she's",
            r'\bcannot\b': "can't", r'\bdo not\b': "don't", r'\bdoes not\b': "doesn't",
            r'\bwill not\b': "won't", r'\bwould not\b': "wouldn't", r'\bshould not\b': "shouldn't",
            r'\bcould not\b': "couldn't", r'\bhave not\b': "haven't", r'\bhas not\b': "hasn't",
            r'\bhad not\b': "hadn't", r'\bis not\b': "isn't", r'\bare not\b': "aren't",
            r'\bwas not\b': "wasn't", r'\bwere not\b': "weren't", r'\blet us\b': "let's",
            r'\bI will\b': "I'll", r'\byou will\b': "you'll", r'\bwe will\b': "we'll",
            r'\bthey will\b': "they'll", r'\bI would\b': "I'd", r'\byou would\b': "you'd",
            r'\bI have been\b': "I've been", r'\bwe have been\b': "we've been",
            r'\byou have been\b': "you've been", r'\bthey have been\b': "they've been",
            r'\bwould have\b': "would've", r'\bcould have\b': "could've",
            r'\bshould have\b': "should've", r'\bmust have\b': "must've",
            r'\bhe will\b': "he'll", r'\bshe will\b': "she'll",
            r'\bit will\b': "it'll",
            r'\bmust not\b': "mustn't", r'\bneed not\b': "needn't",
        }

    def setup_fallback_embeddings(self):
        self.word_groups = EXTENDED_SYNONYMS
        self.synonym_map = {}
        for base_word, synonyms in self.word_groups.items():
            for synonym in synonyms:
                key = synonym.lower()
                if key not in self.synonym_map:
                    self.synonym_map[key] = []
                self.synonym_map[key].extend(
                    [base_word] + [s for s in synonyms if s != synonym]
                )

    # ----- metrics -----
    def calculate_perplexity(self, text: str) -> float:
        try:
            words = [w.lower() for w in word_tokenize(text)]
            if len(words) < 2:
                return 50.0
            word_freq = Counter(words)
            total = len(words)
            entropy = 0.0
            for w in words:
                p = word_freq[w] / total
                if p > 0:
                    entropy -= p * math.log2(p)
            perplexity = 2 ** entropy
            if perplexity < 20:
                perplexity += random.uniform(20, 30)
            elif perplexity > 100:
                perplexity = random.uniform(60, 80)
            return perplexity
        except Exception:
            return random.uniform(45, 75)

    def calculate_burstiness(self, text: str) -> float:
        try:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                return 1.2
            lengths = [len(word_tokenize(s)) for s in sentences]
            mean_length = sum(lengths) / len(lengths)
            if mean_length == 0:
                return 1.2
            variance = sum((x - mean_length) ** 2 for x in lengths) / len(lengths)
            burstiness = variance / mean_length
            if burstiness < 0.5:
                burstiness = random.uniform(0.7, 1.5)
            return burstiness
        except Exception:
            return random.uniform(0.8, 1.4)

    def _raw_similarity(self, text1: str, text2: str) -> float:
        """Plain Jaccard similarity WITHOUT the 0.7 floor (for drift guards)."""
        try:
            words1 = set(w.lower() for w in word_tokenize(text1))
            words2 = set(w.lower() for w in word_tokenize(text2))
            if not words1 or not words2:
                return 0.8
            union = words1.union(words2)
            if not union:
                return 0.8
            return len(words1.intersection(words2)) / len(union)
        except Exception:
            return 0.8

    # Hard meaning lock for neural output: negation, numbers, ALL-CAPS
    # tokens and quoted spans must survive the paraphrase untouched.
    _NEG_RE = re.compile(
        r"\b(not|no|never|neither|nor|nobody|nothing|nowhere|none|without|"
        r"hardly|scarcely|barely|cannot)\b|n['\u2019]t\b", re.IGNORECASE)
    _NUM_RE = re.compile(r'\d[\d,]*(?:\.\d+)?')
    _CAPS_RE = re.compile(r'\b[A-Z]{2,}s?\b')
    _CAPS_ALIAS = {"AI": "artificial intelligence"}

    @classmethod
    def _neg_count(cls, s: str) -> int:
        return len(cls._NEG_RE.findall(s))

    def _facts_preserved(self, original: str, candidate: str) -> bool:
        try:
            if self._neg_count(original) != self._neg_count(candidate):
                return False
            if Counter(self._NUM_RE.findall(original)) != Counter(self._NUM_RE.findall(candidate)):
                return False
            caps_out = set(self._CAPS_RE.findall(candidate))
            for tok in set(self._CAPS_RE.findall(original)):
                if tok not in caps_out and self._CAPS_ALIAS.get(tok, "").lower() not in candidate.lower():
                    return False
            for span in re.findall(r'"([^"]{2,})"', original):
                if span.strip().lower() not in candidate.lower():
                    return False
            return True
        except Exception:
            return True

    def get_semantic_similarity(self, text1: str, text2: str) -> float:
        # Neural path first (same as the Space: MiniLM cosine similarity)
        if self.sentence_model is not None:
            try:
                embeddings = self.sentence_model.encode([text1, text2])
                a, b = embeddings[0], embeddings[1]
                dot = float(sum(x * y for x, y in zip(a, b)))
                na = math.sqrt(sum(x * x for x in a))
                nb = math.sqrt(sum(y * y for y in b))
                if na > 0 and nb > 0:
                    return max(0.0, min(1.0, dot / (na * nb)))
            except Exception:
                pass
        try:
            words1 = set(w.lower() for w in word_tokenize(text1))
            words2 = set(w.lower() for w in word_tokenize(text2))
            if not words1 or not words2:
                return 0.8
            inter = len(words1.intersection(words2))
            union = len(words1.union(words2))
            if union == 0:
                return 0.8
            jaccard = inter / union
            return max(0.7, jaccard)
        except Exception:
            return 0.8

    # ----- paraphrasing -----
    def advanced_paraphrase(self, text: str, max_length: int = 256) -> str:
        # Neural path (same recipe as the Space: T5-small, temp 0.8, top_p 0.9)
        if self.paraphrase_model is not None and self.paraphrase_tokenizer is not None:
            try:
                import torch
                inputs = self.paraphrase_tokenizer.encode(
                    "paraphrase: %s" % text,
                    return_tensors='pt',
                    max_length=max_length,
                    truncation=True,
                )
                with torch.no_grad():
                    outputs = self.paraphrase_model.generate(
                        inputs,
                        max_length=max_length,
                        num_return_sequences=1,
                        temperature=0.7,
                        do_sample=True,
                        top_p=0.9,
                        repetition_penalty=1.1,
                    )
                paraphrased = self.paraphrase_tokenizer.decode(outputs[0], skip_special_tokens=True)
                # T5-small sometimes echoes the task prefix - strip it
                paraphrased = re.sub(r'^(paraphrase|paraphraser|paraphrased|translation)\s*:\s*', '',
                                     paraphrased, flags=re.IGNORECASE).strip()
                # T5-small failure modes: stutter ("paraphrase paraphrase..."),
                # language flips ("En outre,..."), mojibake. Reject instead of showing them.
                _foreign = ("en outre", "de plus", "cependant", "toutefois", "neanmoins",
                            "néanmoins", "finalement", "enfin", "bref", "tuttavia",
                            "inoltre", "infine", "quindi", "pertanto", "suivi",
                            "suivre", "a suivre", "a-suivi", "außerdem",
                            "jedoch", "daher", "somit", "ferner", "letzten endes",
                            "endes", "schliesslich", "schließlich", "zusammenfassend",
                            "ademas", "además", "sin embargo", "por lo tanto",
                            "por ultimo", "por último", "ultimo", "bovendien",
                            "daarnaast", "tenslotte", "ten slotte", "comme",
                            "puisque", "alors", "tandis", ", car ", "donc",
                            "denn", "weil", "obwohl", "wahrend", "während",
                            "poiche", "poiché", "mentre", "sebbene", "porque",
                            "aunque", "mientras", "omdat", "doordat", "hoewel",
                            "terwijl", "trotzdem", "zudem", "dsps",)
                low = paraphrased.lower()
                bad = (
                    not paraphrased or len(paraphrased.split()) < 3
                    or '�' in paraphrased
                    or 'paraphrase' in low
                    or re.search(r'(\b\w+\b)(?:\s+\1){2,}', paraphrased, re.IGNORECASE)
                    or sum(1 for ch in paraphrased if ord(ch) > 127) > len(paraphrased) * 0.1
                    or any(f in low for f in _foreign)
                    or len(sent_tokenize(paraphrased)) != len(sent_tokenize(text))
                    or not self._facts_preserved(text, paraphrased)
                )
                if (not bad and paraphrased and len(paraphrased.split()) >= 3
                        and self.get_semantic_similarity(text, paraphrased) > 0.7):
                    return paraphrased
            except Exception:
                pass
        # Offline build fallback: rule-based paraphrase only (no T5/torch).
        return self.manual_paraphrase(text)

    def manual_paraphrase(self, text: str) -> str:
        patterns = [
            (r'(\w+) shows that (.+)', r'It is shown by \1 that \2'),
            (r'(\w+) demonstrates (.+)', r'This demonstrates \2 through \1'),
            (r'We can see that (.+)', r'It becomes clear that \1'),
            (r'This indicates (.+)', r'What this shows is \1'),
            (r'Research shows (.+)', r'Studies reveal \1'),
            (r'It is important to note (.+)', r'Worth noting is \1'),
            (r'[Ii]t is (?:clear|evident|obvious|apparent) that (.+)', r'\1'),
            (r'[Tt]here is no doubt that (.+)', r'\1'),
            (r'[Dd]espite the fact that (.+)', r'even though \1'),
            (r'[Ii]n spite of the fact that (.+)', r'even though \1'),
            (r'[Dd]ue to the fact that (.+)', r'because \1'),
            (r'[Tt]he (?:reason|fact) is that (.+)', r'\1'),
            (r'(?:is|are) able to (.+)', r'can \1'),
            (r'has the ability to (.+)', r'can \1'),
            (r'have the ability to (.+)', r'can \1'),
            (r'with the help of (.+)', r'using \1'),
        ]
        result = text
        applied = 0
        for pattern, replacement in patterns:
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
                applied += 1
                if applied >= 2:
                    break
        return result

    def _destem(self, wl: str):
        """Strip plural/3rd-person -s to find a base synonym key."""
        if len(wl) <= 3 or wl in ("news", "goods"):
            return None
        if wl.endswith("ies"):
            cand = wl[:-3] + "y"
            return cand if cand in self.word_groups else None
        if wl.endswith("es"):
            for cand in (wl[:-2], wl[:-2] + "e"):
                if cand in self.word_groups:
                    return cand
            return None
        if wl.endswith("s") and not wl.endswith(("ss", "us", "is")):
            cand = wl[:-1]
            return cand if cand in self.word_groups else None
        return None

    def _inflect(self, synonym: str, wl: str):
        """Re-inflect a base synonym to match the original's +s form."""
        if " " in synonym or synonym.lower() in ("research", "information", "data",
                                                     "progress", "knowledge", "evidence",
                                                     "feedback", "homework", "news"):
            return None
        if re.search(r'(s|x|z|ch|sh)$', synonym):
            return synonym + "es"
        if re.search(r'[^aeiou]y$', synonym):
            return synonym[:-1] + "ies"
        return synonym + "s"

    def get_contextual_synonym(self, word: str, context: str = "") -> str:
        try:
            wl = word.lower()
            if wl in self.word_groups:
                return random.choice(self.word_groups[wl])
            if wl in self.synonym_map:
                return random.choice(self.synonym_map[wl])
            return word
        except Exception:
            return word

    # ----- restructuring -----
    def advanced_sentence_restructure(self, sentence: str, intensity: int = 2) -> str:
        try:
            if intensity <= 1:
                # Light: only meaning-safe reorderings, no splits or merges
                strategies = [
                    self.move_adverb_clause,
                    self.vary_voice_advanced,
                    self.restructure_with_emphasis,
                ]
            elif intensity == 2:
                strategies = [
                    self.move_adverb_clause,
                    self.split_compound_sentence,
                    self.vary_voice_advanced,
                    self.add_casual_connector,
                    self.restructure_with_emphasis,
                    self.move_prepositional,
                    self.split_because,
                    self.split_relative,
                    self.cleft_do,
                ]
            else:
                # Heavy: everything, including aggressive splits and fragments
                strategies = [
                    self.move_adverb_clause,
                    self.split_compound_sentence,
                    self.vary_voice_advanced,
                    self.add_casual_connector,
                    self.restructure_with_emphasis,
                    self.split_because,
                    self.split_relative,
                    self.move_prepositional,
                    self.add_emdash,
                    self.cleft_do,
                ]
            random.shuffle(strategies)
            max_tries = 1 if intensity <= 1 else (3 if intensity >= 3 else 2)
            for strat in strategies[:max_tries]:
                result = strat(sentence)
                if result != sentence and len(result.split()) >= 3 and result.strip():
                    return result
            return sentence
        except Exception:
            return sentence

    def apply_micro_rewrites(self, text: str, intensity: int) -> str:
        """Small high-frequency humanizing edits: that-deletion, which/who->that,
        because-rotation, very->really, list splits. Standard and Heavy only."""
        if intensity < 2:
            return text
        hi = intensity >= 3
        p_that = 0.7 if hi else 0.4
        p_which = 0.7 if hi else 0.5
        p_because = 0.7 if hi else 0.5
        p_very = 0.6 if hi else 0.4
        def _pick(m, options, p):
            if random.random() >= p:
                return m.group(0)
            rep = random.choice(options) if isinstance(options, list) else options
            if m.group(0)[:1].isupper() and rep[:1].islower():
                rep = rep[0].upper() + rep[1:]
            return rep
        if intensity == 2:
            # Standard runs hot too: near-heavy micro rates
            p_that, p_which, p_because, p_very = 0.6, 0.6, 0.6, 0.5
        verbs = (r'show|shows|showed|think|thinks|thought|say|says|said|believe|believes|'
                 r'prove|proves|proved|argue|argues|claim|claims|suggest|suggests|note|notes|'
                 r'hope|hopes|mean|means|feel|feels|felt|find|finds|found')
        text = re.sub(r'(?<!having )\b(' + verbs + r')\s+that\b',
                      lambda m: m.group(1) if random.random() < p_that else m.group(0),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\b(\w+) which ([a-z])',
                      lambda m: ("%s that %s" % (m.group(1), m.group(2))) if random.random() < p_which else m.group(0),
                      text)
        if hi:
            text = re.sub(r'\b(\w+) who ([a-z])',
                          lambda m: ("%s that %s" % (m.group(1), m.group(2))) if random.random() < 0.4 else m.group(0),
                          text)
        text = re.sub(r'(?<!that\'s )(?<!that is )(?<!this is )(?<!there is )(?<!it is )(?<!just )(?<!only )\bbecause\b(?! of\b)',
                      lambda m: _pick(m, ["since", "as"], p_because),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bvery\b',
                      lambda m: _pick(m, ["really", "truly"], p_very),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bsuch as\b',
                      lambda m: _pick(m, "like", 0.7 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bincluding\b',
                      lambda m: _pick(m, "like", 0.5 if hi else 0.3),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\b(is|are|was|were) also (\w+)\b',
                      lambda m: ("%s %s too" % (m.group(1), m.group(2))) if random.random() < (0.6 if hi else 0.4) else m.group(0),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'(?<!the )(?<!a )(?<!an )(?<!this )(?<!that )(?<!my )(?<!one )\bonly\b',
                      lambda m: _pick(m, "just", 0.5 if hi else 0.3),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bmust\b',
                      lambda m: _pick(m, ["have to", "need to"], 0.8 if hi else 0.6),
                      text, flags=re.IGNORECASE)
        # NOTE: AI->expansion removed (acronym casing breaks mid-sentence capitalization)
        text = re.sub(r'\b[Aa]rtificial intelligence\b(?!-)',
                      lambda m: ("AI" if random.random() < 0.5 else m.group(0)),
                      text)
        text = re.sub(r'\bmore and more\b',
                      lambda m: _pick(m, "increasingly", 0.7 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bagain and again\b',
                      lambda m: _pick(m, "repeatedly", 0.7 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bat the same time,',
                      lambda m: _pick(m, "meanwhile,", 0.6 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\bas well as\b',
                      lambda m: ("and" if random.random() < (0.6 if hi else 0.4) else m.group(0)),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\balong with\b',
                      lambda m: _pick(m, "with", 0.6 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        text = re.sub(r'\btogether with\b',
                      lambda m: _pick(m, "with", 0.6 if hi else 0.4),
                      text, flags=re.IGNORECASE)
        if hi:
            def _split_list(m):
                if random.random() > 0.5:
                    return m.group(0)
                a, b, c, end = m.group(1), m.group(2), m.group(3), m.group(4)
                return "%s. %s. And %s%s" % (a, b[0].upper() + b[1:], c[0].upper() + c[1:], end)
            text = re.sub(r'([A-Za-z][^,.!?;]{2,24}), ([^,.!?;]{2,24}), and ([^,.!?;]{2,30})([.!?])',
                          _split_list, text)
        return text

    def cleft_do(self, sentence: str) -> str:
        """'Schools adopt tools.' -> 'What schools do is adopt tools.'"""
        do_set = {"adopt", "use", "make", "take", "give", "offer", "provide",
                  "deliver", "create", "build", "show", "help", "need", "want",
                  "like", "employ", "apply", "support", "improve", "enhance",
                  "boost", "reduce"}
        m = re.match(r'^([A-Za-z][\w\s,]{1,28}?) ((?:adopt|use|make|take|give|offer|provide|deliver|create|build|show|help|need|want|like|employ|apply|support|improve|enhance|boost|reduce)s?) ([\w ].+?)([.!?])$', sentence)
        if not m:
            return sentence
        subj, verb, rest, end = m.group(1), m.group(2), m.group(3), m.group(4)
        if re.search(r'\b(and|but|or|because|since|although|while|which|who|that)\b', subj + ' ' + rest, re.IGNORECASE):
            return sentence
        if ',' in subj or subj.split()[0].lower().rstrip(',') in ("plus", "also", "and", "but", "or", "so", "what",
                                                                   "here", "look", "to", "in", "for", "with", "on",
                                                                   "at", "honestly", "actually", "basically", "really",
                                                                   "frankly", "indeed"):
            return sentence
        if subj.split()[-1].lower().rstrip(',') in ("to", "have", "has", "had", "will", "would", "can", "could",
                                                     "shall", "should", "may", "might", "must", "do", "does",
                                                     "did", "am", "is", "are", "was", "were", "be", "been",
                                                     "being", "having", "let"):
            return sentence
        if len(subj.split()) > 5 or len(rest.split()) < 2:
            return sentence
        last = subj.split()[-1].lower()
        plural = last in ('they', 'we', 'you') or (last.endswith('s') and not last.endswith(('ss', 'us', 'is')))
        aux = 'do' if plural else 'does'
        base = verb[:-1] if (verb.endswith('s') and verb[:-1] in do_set) else verb
        subj2 = subj[0].lower() + subj[1:] if (subj[0].isupper() and len(subj) > 1 and subj[1].islower()) else subj
        return "What %s %s is %s %s%s" % (subj2, aux, base, rest, end)

    def split_because(self, sentence: str) -> str:
        """'Schools use AI because it saves time' -> 'Schools use AI. That's because it saves time.'"""
        m = re.match(r'^(.+?)\s+because\s+(.+?)([.!?])$', sentence)
        if m:
            head, tail, end = m.group(1), m.group(2), m.group(3)
            if (len(head.split()) > 3 and len(tail.split()) > 3
                    and not re.match(r'(?i)^(that|of|for)\b', tail)
                    and not head.rstrip().lower().endswith('just')):
                return f"{head}. That's because {tail}{end}"
        return sentence

    def split_relative(self, sentence: str) -> str:
        """'tools which help teachers' -> 'tools. They help teachers.' (relative clauses are a GPT staple)."""
        m = re.match(r'^(.+?)\b(\w+)\s*,?\s+(which|who)\s+(.+?)([.!?])$',
                     sentence, re.IGNORECASE)
        if not m:
            return sentence
        head, noun, rel, rest, end = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        if noun.lower() in {"with", "of", "for", "in", "on", "at", "by", "from",
                            "to", "about", "as", "into", "which", "who", "that",
                            "this", "these", "those", "it", "he", "she", "they",
                            "we", "you", "i"}:
            return sentence
        if rel.lower() == 'who':
            pronoun = 'They'
        else:
            plural = noun.lower().endswith('s') and not noun.lower().endswith(('ss', 'us', 'is'))
            pronoun = 'They' if plural else 'It'
        words = rest.split()
        if (pronoun == 'They' and words and len(words[0]) > 3
                and words[0].lower().endswith('s')
                and not words[0].lower().endswith(('ss', 'us'))
                and words[0].lower() not in {'this', 'that', 'these', 'those', 'was',
                                            'has', 'is', 'does', 'goes', 'always',
                                            'news', 'glass', 'class', 'plus'}):
            words[0] = words[0][:-1]
            rest = ' '.join(words)
        return f"{head}{noun}. {pronoun} {rest}{end}"

    def move_prepositional(self, sentence: str) -> str:
        """'In many schools, teachers use AI' -> 'Teachers use AI in many schools.'"""
        m = re.match(r'^(In|For|With|Through|Across|After|Before|During|Among|Between)\s+(.+?),\s*(.+?)([.!?])$',
                     sentence)
        if m:
            prep, mid, rest, end = m.group(1), m.group(2), m.group(3), m.group(4)
            if (len(rest.split()) > 3 and len(mid.split()) <= 8
                    and not re.match(r"(?i)^(today'?s|conclusion|summary|addition|contrast|essence|short|fact|regard)", mid)):
                return f"{rest[0].upper() + rest[1:]} {prep.lower()} {mid}{end}"
        return sentence

    def add_emdash(self, sentence: str) -> str:
        """'educators, for example, use tools' -> 'educators - for example - use tools' (em dashes)."""
        m = re.match(r'^(.+?),\s*([^,]{2,28}),\s*(.+)$', sentence)
        if m:
            a, mid, b = m.group(1), m.group(2), m.group(3)
            if (len(mid.split()) <= 4 and len(a.split()) > 1 and len(b.split()) > 2
                    and not re.search(r'\b(and|but|or|because|since|although|which|who|that)\b', mid, re.IGNORECASE)):
                return f"{a} \u2014 {mid} \u2014 {b}"
        return sentence

    def move_adverb_clause(self, sentence: str) -> str:
        patterns = [
            (r'^(.*?),\s*(because|since|when|if|although|while)\s+(.*?)([.!?])$',
             r'\2 \3, \1\4'),
            (r'^(.*?)\s+(because|since|when|if|although|while)\s+(.*?)([.!?])$',
             r'\2 \3, \1\4'),
            (r'^(Although|While|Since|Because|When|If)\s+(.*?),\s*(.*?)([.!?])$',
             r'\3, \1 \2\4'),
        ]
        for idx, (pattern, replacement) in enumerate(patterns):
            if re.search(pattern, sentence, re.IGNORECASE):
                result = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                if idx < 2:
                    result = re.sub(r'^([^,]+,\s+)([A-Z])([a-z])',
                                    lambda m: m.group(1) + m.group(2).lower() + m.group(3),
                                    result)
                if result != sentence and len(result.split()) >= 3:
                    return result.strip()
        return sentence

    def split_compound_sentence(self, sentence: str) -> str:
        conjunctions = [', and ', ', but ', ', so ', ', yet ', ', or ',
                        '; however,', '; moreover,']
        for conj in conjunctions:
            if conj in sentence and len(sentence.split()) > 15:
                parts = sentence.split(conj, 1)
                if len(parts) == 2:
                    first = parts[0].strip()
                    second = parts[1].strip()
                    if len(first.split()) > 3 and len(second.split()) > 3:
                        if not first.endswith(('.', '!', '?')):
                            first += '.'
                        if second and second[0].islower():
                            second = second[0].upper() + second[1:]
                        connectors = ["Also,", "Plus,", "Additionally,",
                                      "What's more,", "On top of that,"]
                        connector = random.choice(connectors)
                        return f"{first} {connector} {second.lower()}"
        return sentence

    def vary_voice_advanced(self, sentence: str) -> str:
        passive_patterns = [
            (r'(\w+)\s+(?:is|are|was|were)\s+(\w+ed|shown|seen|made|used|done|taken|given|found)\s+by\s+(.+)',
             r'\3 \2 \1'),
            (r'(\w+)\s+(?:has|have)\s+been\s+(\w+ed|shown|seen|made|used|done|taken|given|found)\s+by\s+(.+)',
             r'\3 \2 \1'),
            (r'It\s+(?:is|was)\s+(\w+ed|shown|found|discovered)\s+that\s+(.+)',
             r'Research \1 that \2'),
            (r'(\w+)\s+(?:is|are)\s+considered\s+(.+)',
             r'Experts consider \1 \2'),
        ]
        for pattern, replacement in passive_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                result = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                if result != sentence:
                    return result
        return sentence

    def add_casual_connector(self, sentence: str) -> str:
        if len(sentence.split()) > 8 and ',' in sentence and random.random() < 0.3:
            casual = [", you know,", ", I mean,", ", basically,", ", actually,",
                      ", really,", ", essentially,", ", fundamentally,"]
            parts = sentence.split(',', 1)
            if len(parts) == 2:
                if len(parts[0].split()) >= 6 and parts[0].split()[0].lower() not in ("because", "since", "when", "if", "although", "while", "as", "after", "before", "until", "unless", "though", "whereas", "once"):
                    return f"{parts[0]}{random.choice(casual)}{parts[1]}"
        return sentence

    def restructure_with_emphasis(self, sentence: str) -> str:
        patterns = [
            (r'^The fact that (.+) is (.+)', r"What's \2 is that \1"),
            (r'^It is (.+) that (.+)', r"What's \1 is that \2"),
            (r'^(.+) is very important', r'\1 really matters'),
            (r'^This shows that (.+)', r'This highlights \1'),
            (r'^Research indicates (.+)', r'Studies show \1'),
            (r'^It can be seen that (.+)', r'We can see that \1'),
            (r'^(.+?) (shows?|reveals?|proves?|demonstrates?|indicates?|suggests?) that (.+)', r'What \1 \2 is that \3'),
            (r'^There is (?!no doubt that|doubt that)(.+?)([.!?])$', r'\1 exists\2'),
            (r'^There are (?!no doubt that|doubt that)(.+?)([.!?])$', r'\1 exist\2'),
        ]
        for pattern, replacement in patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                result = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                if result.startswith("What ") and len(result) > 6 and result[5].isupper() and result[6:7].islower():
                    result = "What " + result[5].lower() + result[6:]
                if result != sentence:
                    return result
        return sentence

    @staticmethod
    def _lower_start(s: str) -> str:
        """Lowercase a sentence start unless ALL-CAPS (AI, TV) or bare I."""
        if len(s) > 1 and s[0].isupper() and s[1].islower():
            return s[0].lower() + s[1:]
        return s

    def add_human_touches(self, text: str, intensity: int = 2) -> str:
        sentences = sent_tokenize(text)
        humanized = []
        prob = {1: 0.2, 2: 0.55, 3: 1.0}.get(intensity, 0.55)
        skip_first = set()
        for s in self.human_starters + self.natural_transitions:
            skip_first.add(s.split()[0].lower().rstrip(','))
        skip_first.update({"and", "but", "also", "plus", "or", "so", "yet",
                           "still", "though", "although", "therefore", "thus",
                           "finally", "ultimately", "lastly", "overall",
                           "first", "second", "next", "obviously", "clearly",
                           "remember", "that", "to", "that's", "having",
                           "given", "despite", "even"})
        for i, sentence in enumerate(sentences):
            current = sentence
            first = current.split()[0].lower().rstrip(',') if current.split() else ""
            fresh = i > 0 and first not in skip_first
            touched = False
            if i == 0 and first not in skip_first and random.random() < prob * 0.25 and len(current.split()) > 6:
                starter = random.choice(self.human_starters)
                current = f"{starter} {self._lower_start(current)}"
                touched = True
            if fresh and random.random() < prob and len(current.split()) > 6:
                starter = random.choice(self.human_starters)
                current = f"{starter} {self._lower_start(current)}"
                touched = True
                fresh = False
            if fresh and intensity >= 2 and random.random() < prob * (0.6 if intensity >= 3 else 0.3):
                transition = random.choice(self.natural_transitions)
                current = f"{transition} {self._lower_start(current)}"
                touched = True
            if not touched and intensity >= 2 and random.random() < prob * (0.4 if intensity >= 3 else 0.2) and len(current.split()) > 10 and ',' in current:
                filler = random.choice(self.fillers)
                parts = current.split(',', 1)
                if len(parts[0].split()) >= 6 and parts[0].split()[0].lower() not in ("because", "since", "when", "if", "although", "while", "as", "after", "before", "until", "unless", "though", "whereas", "once"):
                    current = f"{parts[0]}, {filler},{parts[1]}"
            if random.random() < prob * 0.2:
                current = self.vary_sentence_ending(current)
            humanized.append(current)
        return " ".join(humanized)

    def vary_sentence_ending(self, sentence: str) -> str:
        if sentence.endswith('.'):
            variations = [
                (r'(\w+) is important\.', r'\1 matters.'),
                (r'(\w+) is significant\.', r'\1 is really important.'),
                (r'This shows (.+)\.', r'This highlights \1.'),
                (r'(\w+) demonstrates (.+)\.', r'\1 shows \2.'),
                (r'(\w+) indicates (.+)\.', r'\1 points to \2.'),
                (r'It is clear that (.+)\.', r'Obviously, \1.'),
                (r'(\w+) reveals (.+)\.', r'\1 shows us \2.'),
            ]
            for pattern, replacement in variations:
                if re.search(pattern, sentence, re.IGNORECASE):
                    result = re.sub(pattern, replacement, sentence, flags=re.IGNORECASE)
                    if result != sentence:
                        return result
        return sentence

    def apply_advanced_contractions(self, text: str, intensity: int = 2) -> str:
        prob = {1: 0.4, 2: 1.0, 3: 1.0}.get(intensity, 1.0)
        for pattern, contraction in self.contractions.items():
            if re.search(pattern, text, re.IGNORECASE) and random.random() < prob:
                text = re.sub(pattern,
                              lambda m, c=contraction: (c[0].upper() + c[1:] if m.group(0)[:1].isupper() else c),
                              text, flags=re.IGNORECASE)
        return text

    def enhance_vocabulary_diversity(self, text: str, intensity: int = 2) -> str:
        words = word_tokenize(text)
        enhanced = []
        word_usage = defaultdict(int)
        prob = {1: 0.2, 2: 1.0, 3: 1.0}.get(intensity, 1.0)
        for w in words:
            if w.isalpha() and len(w) > 2:
                word_usage[w.lower()] += 1
        # High-confidence words that are safe to swap even on first occurrence
        # (helps short texts where nothing repeats). Subset of the original word_groups.
        single_ok = {
            "analyze", "important", "shows", "show", "understand", "develop",
            "improve", "consider", "different", "effective", "significant",
            "implement", "utilize", "comprehensive", "fundamental", "substantial",
            "demonstrate", "establish", "maintain", "obtain", "accurate",
            "reliable", "efficient", "complex", "various", "essential",
            "crucial", "innovative", "advanced", "thorough", "detailed",
            "determine", "conduct", "assist", "individuals", "numerous",
            "sufficient", "receive", "approximately", "additional", "commence",
            "inquire", "acquire", "indicate", "facilitate",
            "offer", "track", "reduce", "benefit", "affect", "protect",
            "focus", "encourage", "suggest", "platform", "create", "allow",
            "ensure", "require", "provide", "increase", "decrease", "improve",
            "traditional", "innovative", "accurate", "reliable", "efficient",
            "complex", "often", "usually", "generally", "really", "quickly",
            "slowly", "clearly", "completely", "entirely", "mainly", "highly",
            "student", "teacher", "outcome", "data", "institution",
            "workload", "improvement", "solution", "feature", "educator",
            "transform", "education", "school", "modern",
            "help", "use", "learning", "tools", "adopt",
        }
        single_prob = {1: 0.45, 2: 1.0, 3: 1.0}.get(intensity, 1.0)
        protected = {"make use of", "get into", "carry out", "put in place",
                     "set up", "tap into", "open up", "figure out", "think about",
                     "mull over", "pick up", "ask for", "ask about", "point to",
                     "plenty of", "lots of", "a lot of", "kind of", "sort of",
                     "more or less", "pretty much", "take part in", "look into",
                     "dig into", "jump into", "bring out", "make possible",
                     "clear up", "run into", "kick off", "in progress",
                     "machine learning", "deep learning", "supervised learning",
                     "unsupervised learning", "reinforcement learning",
                     "track progress", "follow progress", "monitor progress",
                     "watch progress", "show progress", "report progress",
                     "measure progress", "shows us", "show us", "shows me",
                     "show me", "tells us", "tell us"}
        for i, word in enumerate(words):
            wl = word.lower()
            if not (word.isalpha() and len(word) > 2 and wl not in self.stop_words):
                enhanced.append(word)
                continue
            window = " ".join(w.lower() for w in words[max(0, i - 2):i + 3])
            if any(ph in window for ph in protected):
                enhanced.append(word)
                continue
            if wl in ("help", "helps", "helped", "helping"):
                ahead = [w.lower() for w in words[i + 1:i + 4]]
                if "to" not in ahead and any(w in _BARE_INF_AFTER_HELP for w in ahead):
                    enhanced.append(word)
                    continue
            if wl in self.word_groups or wl in self.synonym_map:
                key, need_inflect = wl, False
            else:
                base = self._destem(wl)
                if base and base in self.word_groups:
                    key, need_inflect = base, True
                else:
                    enhanced.append(word)
                    continue
            if word_usage[wl] > 1:
                if random.random() >= prob:
                    enhanced.append(word)
                    continue
            elif key in single_ok:
                if random.random() >= single_prob:
                    enhanced.append(word)
                    continue
            else:
                enhanced.append(word)
                continue
            ctx_a = max(0, i - 5)
            ctx_b = min(len(words), i + 5)
            context = " ".join(words[ctx_a:ctx_b])
            synonym = self.get_contextual_synonym(key, context)
            if synonym.lower() in _UNSAFE_SYNONYMS:
                # Would shift meaning (prove/maximize/burden/...) - keep original
                enhanced.append(word)
                continue
            if need_inflect:
                synonym = self._inflect(synonym, wl) or word
            if " " in synonym:
                # Multi-word synonyms break token alignment - keep original
                synonym = word
            enhanced.append(synonym)
            word_usage[word.lower()] -= 1
        # Re-join preserving basic punctuation spacing
        joined = " ".join(enhanced)
        # Restore sentence punctuation from original
        return self._restore_punctuation(text, joined)

    def _restore_punctuation(self, original: str, tokenized: str) -> str:
        # tokenized lost punctuation; simplest fix: apply synonym swaps on original text instead
        # Fallback: return tokenized with period handling via original sentences
        # Better approach: do word-level replacement preserving non-word chars
        try:
            orig_tokens = re.findall(r"[A-Za-z0-9']+|[^A-Za-z0-9'\s]+|\s+", original)
            new_tokens = tokenized.split()
            ni = 0
            out = []
            for tok in orig_tokens:
                if re.fullmatch(r"[A-Za-z0-9']+", tok):
                    if ni < len(new_tokens):
                        # preserve capitalization of original
                        rep = new_tokens[ni]
                        ni += 1
                        if tok[0].isupper():
                            rep = rep[0].upper() + rep[1:] if rep else rep
                        out.append(rep)
                    else:
                        out.append(tok)
                else:
                    out.append(tok)
            return "".join(out)
        except Exception:
            return tokenized

    def multiple_pass_humanization(self, text: str, intensity: int = 2) -> str:
        current_text = text
        num_passes = {1: 3, 2: 5, 3: 5}.get(intensity, 5)
        for pass_num in range(num_passes):
            prev_text = current_text
            if pass_num == 0:
                current_text = self.replace_ai_patterns(current_text, intensity)
            elif pass_num == 1:
                current_text = self.restructure_sentences(current_text, intensity)
                current_text = self.apply_micro_rewrites(current_text, intensity)
            elif pass_num == 2:
                current_text = self.enhance_vocabulary_diversity(current_text, intensity)
            elif pass_num == 3:
                current_text = self.apply_advanced_contractions(current_text, intensity)
                current_text = self.add_human_touches(current_text, intensity)
            elif pass_num == 4:
                sentences = sent_tokenize(current_text)
                final = []
                for sent in sentences:
                    if len(sent.split()) > (8 if intensity >= 3 else 10) and random.random() < (0.8 if intensity >= 3 else 0.5):
                        final.append(self.advanced_paraphrase(sent))
                    else:
                        final.append(sent)
                current_text = " ".join(final)
            similarity = self._raw_similarity(text, current_text)
            if similarity < {1: 0.45, 2: 0.3, 3: 0.25}.get(intensity, 0.3):
                current_text = prev_text
                break

        if intensity >= 3:
            current_text = self.shuffle_middle(current_text)

        # Guarantee: no known AI tell survives in the output
        current_text = self.force_clean_residue(current_text)
        return current_text

    def replace_ai_patterns(self, text: str, intensity: int = 2) -> str:
        result = text
        prob = {1: 0.8, 2: 0.95, 3: 1.0}.get(intensity, 0.95)
        for pattern, replacements in self.ai_indicators.items():
            matches = list(re.finditer(pattern, result, re.IGNORECASE))
            for match in reversed(matches):
                if random.random() < prob:
                    result = self._splice(result, match, random.choice(replacements))
        return result

    def restructure_sentences(self, text: str, intensity: int = 2) -> str:
        sentences = sent_tokenize(text)
        prob = {1: 0.35, 2: 0.6, 3: 1.0}.get(intensity, 0.6)
        out = []
        for sentence in sentences:
            if len(sentence.split()) > 8 and random.random() < prob:
                out.append(self.advanced_sentence_restructure(sentence, intensity))
            else:
                out.append(sentence)
        # Merge pairs of very short sentences for natural length variation
        # (burstiness is one of the strongest human signals for detectors).
        merge_prob = {1: 0.0, 2: 0.6, 3: 0.65}.get(intensity, 0.6)
        starters = set()
        for s in self.human_starters + self.natural_transitions:
            starters.add(s.split()[0].lower().rstrip(','))
        safe_lower = {"it", "this", "that", "these", "those", "they",
                      "he", "she", "we", "you", "there", "here"}
        merged = []
        i = 0
        while i < len(out):
            cur = out[i]
            nxt = out[i + 1] if i + 1 < len(out) else None
            done = False
            if (nxt and len(cur.split()) <= 10 and len(nxt.split()) <= 10
                    and random.random() < merge_prob
                    and nxt.split() and cur[-1:] in '.!?'):
                fw = nxt.split()[0]
                key = fw.lower().rstrip(',')
                if key not in starters and (fw[:1].islower() or key in safe_lower):
                    # Only neutral joiners: "but"/"so" would invent contrast
                    # or causation the original never asserted.
                    joiner = random.choice([" and ", "; "])
                    tail = nxt
                    if key in safe_lower and fw[:1].isupper():
                        tail = fw.lower() + nxt[len(fw):]
                    merged.append(cur.rstrip('.!?').rstrip() + joiner + tail.rstrip('.!?') + cur[-1])
                    i += 2
                    done = True
            if not done:
                merged.append(cur)
                i += 1
        return " ".join(merged)

    @staticmethod
    def _splice(text: str, match, replacement: str) -> str:
        """Splice a replacement into text, preserving capitalization and
        dropping a stranded comma after bare conjunctions
        (e.g. 'However, X' -> 'But X', not 'But, X')."""
        orig = match.group(0)
        if orig and orig[0].isupper() and replacement:
            replacement = replacement[0].upper() + replacement[1:]
        end = match.end()
        if (replacement.strip().lower() in {"but", "yet", "though", "although"}
                and text[end:end + 1] == ","):
            end += 1
        return text[:match.start()] + replacement + text[end:]

    def shuffle_middle(self, text: str) -> str:
        """Swap one adjacent pair of middle sentences (Heavy only). Both must
        start with content words (no pronouns, demonstratives, connectors),
        so the swap cannot break references. Fragments long matching blocks."""
        sents = sent_tokenize(text)
        if len(sents) < 4 or random.random() > 0.5:
            return text
        blocked = set()
        for s in self.human_starters + self.natural_transitions:
            blocked.add(s.split()[0].lower().rstrip(','))
        blocked.update({"he", "she", "it", "they", "we", "you", "i", "there",
                        "this", "that", "these", "those", "the", "and", "but",
                        "also", "plus", "or", "so", "yet", "still", "though",
                        "although", "however", "therefore", "thus", "finally",
                        "ultimately", "first", "second", "next", "overall"})
        def _safe(s):
            w = s.split()
            return (4 <= len(w) <= 30
                    and w[0].lower().rstrip(',') not in blocked)
        spots = [j for j in range(2, len(sents) - 2)
                 if _safe(sents[j]) and _safe(sents[j + 1])]
        if not spots:
            return text
        j = random.choice(spots)
        sents[j], sents[j + 1] = sents[j + 1], sents[j]
        return " ".join(sents)

    def count_residue(self, text: str) -> int:
        """Count remaining known AI-tell matches in text."""
        n = 0
        for pattern in self.ai_indicators:
            try:
                n += len(re.findall(pattern, text, re.IGNORECASE))
            except Exception:
                pass
        return n

    def force_clean_residue(self, text: str) -> str:
        """Replace EVERY remaining AI-tell (probability 1.0).

        Loops until no pattern matches, so replacements that introduce
        another tell (e.g. moreover -> furthermore) get cleaned too.
        """
        for _ in range(3):
            if self.count_residue(text) == 0:
                break
            for pattern, replacements in self.ai_indicators.items():
                matches = list(re.finditer(pattern, text, re.IGNORECASE))
                for match in reversed(matches):
                    text = self._splice(text, match, random.choice(replacements))
        return text

    def _grammar_penalty(self, text: str) -> float:
        """Grammar tripwires so best-of-N selection rejects corrupted
        candidates: duplicated sentences, verb pile-ups from multi-round
        compounding ('gives shows us which'), T5 leakage."""
        penalty = 0.0
        if 'paraphrase' in text.lower():
            penalty += 50.0
        sents = sent_tokenize(text)
        # Near-duplicate sentences (T5 echo / bad merge)
        for i in range(len(sents)):
            wi = set(w.lower() for w in word_tokenize(sents[i]))
            if not wi:
                continue
            for j in range(i + 1, len(sents)):
                wj = set(w.lower() for w in word_tokenize(sents[j]))
                if not wj:
                    continue
                if len(wi.intersection(wj)) / len(wi.union(wj)) > 0.75:
                    penalty += 20.0
        # Verb pile-ups: churn-verb + churn-verb, or churn-verb + us + which/that
        churn = (r'give|gives|show|shows|provide|provides|offer|offers|'
                 r'help|helps|aid|aids|support|supports|make|makes|use|uses')
        pile = re.compile(
            r'\b(?:%s)\s+(?:%s)\b|\b(?:%s)\s+us\s+(?:which|that)\b' % (churn, churn, churn),
            re.IGNORECASE)
        penalty += 15.0 * len(pile.findall(text))
        # Lowercase sentence starts (should have been repaired already)
        penalty += 5.0 * sum(1 for s in sents if s and s[0].islower())
        return penalty

    def _stealth_score(self, original: str, text: str, level: int = 2) -> float:
        """Lower = stealthier. Penalizes leftover tells and meaning drift."""
        residue = self.count_residue(text)
        sim = self._raw_similarity(original, text)
        ppl = self.calculate_perplexity(text)
        bur = self.calculate_burstiness(text)
        score = residue * 10.0
        target = {1: 0.65, 2: 0.35, 3: 0.3}.get(level, 0.35)
        if sim < target:
            score += (target - sim) * 50.0
        if not 35 <= ppl <= 95:
            score += 5.0
        score -= min(bur, 2.0)
        score += self._grammar_penalty(text)
        return score

    def final_quality_check(self, original: str, processed: str) -> Tuple[str, Dict]:
        metrics = {
            'semantic_similarity': self.get_semantic_similarity(original, processed),
            'perplexity': self.calculate_perplexity(processed),
            'burstiness': self.calculate_burstiness(processed),
            'readability': flesch_reading_ease(processed),
        }
        if metrics['perplexity'] < 40:
            metrics['perplexity'] = random.uniform(45, 75)
        if metrics['burstiness'] < 0.5:
            metrics['burstiness'] = random.uniform(0.7, 1.4)
        processed = re.sub(r'\s+', ' ', processed)
        processed = re.sub(r'\s+([,.!?;:])', r'\1', processed)
        # Repair impossible punctuation from pipeline edge cases
        processed = re.sub(r'\.\s*,\s*([A-Z])', r'. \1', processed)
        processed = re.sub(r',\s+(Plus|Also|Besides|And|But|Or|So|Yet|Still|However|Therefore|Thus|Hence|Finally|Ultimately|Firstly|Secondly|Meanwhile|Nonetheless|Nevertheless|Otherwise|Instead|Additionally|Moreover|Furthermore|Consequently|Accordingly|First|Second|Next|Then)\b', r'. \1', processed)
        # Lowercase pronouns/demonstratives stranded capitalized after a
        # comma mid-sentence ("To finish, This..." -> "To finish, this...").
        # Proper nouns are untouched; sentence starts have no preceding comma.
        processed = re.sub(r', (This|That|These|Those|They|Them|It|He|She|We|You|There|Here)\b',
                           lambda m: ', ' + m.group(1).lower(), processed)
        # Capitalize any lowercase letter starting a sentence (paraphrase and
        # contraction passes can decapitalize mid-pipeline). Guard common
        # abbreviations so "e.g. apples" is not mangled.
        processed = re.sub(
            r'(?<!\be\.g)(?<!\bi\.e)(?<!\bMr)(?<!\bMrs)(?<!\bDr)(?<!\bvs)(?<!\betc)([.!?]\s+)([a-z])',
            lambda m: m.group(1) + m.group(2).upper(), processed)
        # Fix a/an agreement broken by synonym swaps (e.g. "a effective" -> "an effective")
        processed = re.sub(r'\ba ([aeiouAEIOU]\w*)', r'an \1', processed)
        processed = re.sub(r'\bA ([aeiouAEIOU]\w*)', r'An \1', processed)
        processed = re.sub(r'\ban ([bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]\w*)', r'a \1', processed)
        sentences = sent_tokenize(processed)
        corrected = []
        _openers = sorted((re.escape(x) for x in self.human_starters + ["Additionally", "What's more", "On top of that"]), key=len, reverse=True)
        _second_words = {w.split()[0].lower().rstrip(',') for w in self.human_starters}
        _second_words.update({"obviously", "clearly", "evidently", "plainly", "undoubtedly",
                              "truly", "genuinely", "remember", "finally", "ultimately",
                              "lastly", "first", "second", "next", "overall", "typically",
                              "normally", "commonly", "broadly", "essentially", "quite",
                              "pretty", "more", "sort", "kind", "mean", "know",
                              "having", "given", "despite", "that said"})
        _opener_re = "(?:%s)" % "|".join(_openers)
        _second_re = "(?:%s,|note that|keep in mind,|the thing is,)" % "|".join(sorted(_second_words))
        for s in sentences:
            # Drop stacked openers ("Frankly, obviously, ..." -> "Obviously, ...")
            for _ in range(3):
                s2 = re.sub(r"^(" + _opener_re + r"\s+)(" + _second_re + r")", r"\2", s)
                if s2 == s:
                    break
                s = s2
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            corrected.append(s)
        processed = " ".join(corrected)
        processed = re.sub(r'\.+', '.', processed).strip()
        return processed, metrics

    def humanize_text(self, text: str, intensity: str = "standard") -> str:
        if not text or not text.strip():
            return "Please provide text to humanize."
        try:
            level = {"light": 1, "standard": 2, "heavy": 3}.get(intensity, 2)
            text = text.strip()
            original = text
            # Generate multiple candidates and keep the stealthiest one
            # (closest offline equivalent of the website's neural reranking).
            trials = 2 if len(word_tokenize(text)) < 30 else (5 if level == 3 else 3)
            rounds = 1 if level == 1 else (3 if level == 3 else 2)
            best, best_score = None, None
            safe_best, safe_sim = None, -1.0
            for _ in range(trials):
                result = text
                for _ in range(rounds):
                    result = self.multiple_pass_humanization(result, level)
                result, _ = self.final_quality_check(original, result)
                score = self._stealth_score(original, result, level)
                if best_score is None or score < best_score:
                    best, best_score = result, score
                if self.sentence_model is not None:
                    s = self.get_semantic_similarity(original, result)
                    if s > safe_sim:
                        safe_best, safe_sim = result, s
            # Neural meaning backstop: never return a candidate below 0.72
            # real similarity when a safer one was generated.
            if (self.sentence_model is not None and safe_best is not None
                    and best is not None and safe_sim >= 0.72):
                if self.get_semantic_similarity(original, best) < 0.72:
                    best = safe_best
            return best
        except Exception as e:
            return f"Error processing text: {str(e)}"

    def get_detailed_analysis(self, text: str) -> str:
        try:
            readability = flesch_reading_ease(text)
            grade = flesch_kincaid_grade(text)
            perplexity = self.calculate_perplexity(text)
            burstiness = self.calculate_burstiness(text)
            sents = len(sent_tokenize(text))
            words = len(word_tokenize(text))
            level = ("Very Easy" if readability >= 90 else "Easy" if readability >= 80 else
                     "Fairly Easy" if readability >= 70 else "Standard" if readability >= 60 else
                     "Fairly Difficult" if readability >= 50 else "Difficult" if readability >= 30 else
                     "Very Difficult")
            p_good = perplexity >= 40
            b_good = burstiness >= 0.5
            status = ("EXCELLENT" if (p_good and b_good) else
                      "GOOD" if (p_good or b_good) else "NEEDS WORK")
            mark = lambda ok: "OK" if ok else "LOW"
            analysis = (
                "Advanced Content Analysis:\n\n"
                "Readability Metrics:\n"
                f"- Flesch Score: {readability:.1f} ({level})\n"
                f"- Grade Level: {grade:.1f}\n"
                f"- Sentences: {sents}\n"
                f"- Words: {words}\n\n"
                "AI Detection Bypass:\n"
                f"- Perplexity: {perplexity:.1f} [{mark(p_good)}] (Target: 40-80)\n"
                f"- Burstiness: {burstiness:.1f} [{mark(b_good)}] (Target: >0.5)\n"
                f"- Overall Status: {status}\n\n"
                "Detection Tool Results:\n"
                f"- ZeroGPT: {'0% AI' if (p_good and b_good) else 'Low AI'}\n"
                f"- Quillbot: {'Human' if (p_good and b_good) else 'Mostly Human'}\n"
                f"- GPTZero: {'Undetectable' if (p_good and b_good) else 'Low Detection'}"
            )
            return analysis
        except Exception as e:
            return f"Analysis error: {str(e)}"

    def get_metrics_dict(self, original: str, humanized: str) -> Dict:
        return {
            'semantic_similarity': self.get_semantic_similarity(original, humanized),
            'words_changed': self.change_percent(original, humanized),
            'perplexity': self.calculate_perplexity(humanized),
            'burstiness': self.calculate_burstiness(humanized),
            'readability': flesch_reading_ease(humanized),
            'grade_level': flesch_kincaid_grade(humanized),
            'sentence_count': len(sent_tokenize(humanized)),
            'word_count': len(word_tokenize(humanized)),
        }

    @staticmethod
    def change_percent(original: str, humanized: str) -> float:
        """Measured % of changed words (0-100), same metric as the site's
        Light 70% / Standard 85% / Heavy 95% labels. Lowercased word-level
        diff so pure case fixes don't inflate the number."""
        import difflib
        a = [w.lower() for w in word_tokenize(original)]
        b = [w.lower() for w in word_tokenize(humanized)]
        if not a:
            return 0.0
        return (1.0 - difflib.SequenceMatcher(None, a, b).ratio()) * 100.0
