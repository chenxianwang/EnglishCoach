"""Counting vocabulary in the three units that actually mean different things.

`speaking_vocabulary()` counts **word forms**: "go", "goes", "going" and "went"
are four. That is the right unit for "how much have I said", and the wrong unit
for "how big is my vocabulary" — nobody learns "went" separately from "go".

The published vocabulary-size figures (a 5-year-old knows 4-5k, an educated
adult 15-20k) are counted in two other units, and mixing them up moves the
answer by a factor of two:

    form      go  goes  going  went  goer  ongoing        6
    lemma     go (all inflections folded in) · goer · ongoing
                                                          3
    family    go (inflections AND derivations folded in)   1

This module folds forms down to both. It needs no network and no NLP package:
folding is rule-based and every candidate base is checked against a known-word
set, so we never invent one — "running" becomes "run" only because "run" is a
word, and "sung" stays put because "su" is not.

It also carries a ~2000-word frequency reference (`BAND1`, `BAND2`), which is
what makes the counts diagnostic rather than merely large. Two speakers with
1,400 lemmas each are not equally good if one of them never leaves the first
thousand words of English.

    >>> lx = Lexicon()
    >>> lx.lemma("studies"), lx.family("carefully")
    ('study', 'care')

Accuracy: spot-checked against this library's own transcripts, the lemmatiser
is essentially exact and the family folder lands within a few percent. Family
counts are a good measure and not a precise one; treat the third decimal place
as noise, and the trend across months as real.
"""
import os
import re

# ---------------------------------------------------------------------------
# The frequency reference.
#
# BAND1 is roughly the first 1000 word families of English, BAND2 the second
# thousand — the "K1/K2" split used in vocabulary research. Between them they
# cover about 80% of the running words of ordinary conversation, which is the
# whole point: everything a speaker says beyond them is where their range
# actually shows. These are close approximations of the published lists, hand
# maintained here so the app keeps working with no data download.
# ---------------------------------------------------------------------------
BAND1 = set("""
i me my mine myself you your yours yourself yourselves he him his himself she
her hers herself it its itself we us our ours ourselves they them their theirs
themselves a an the this that these those there here what which who whom whose
when where why how not no none nor neither either both each all some any many
much few more most other another same such own very too also just even still
yet again once always never often sometimes usually ever now then soon already
almost enough quite maybe perhaps indeed instead besides however therefore
otherwise meanwhile anyway okay yeah something anything nothing everything
someone anyone everyone somebody anybody everybody nobody somewhere anywhere
everywhere can could may might must shall should will would cannot going
a about above across act add afraid after again against age ago agree air all
allow almost alone along already also although always among amount and angry
animal another answer any appear apple area arm army around arrive art as ask
at attack aunt autumn away
baby back bad bag ball band bank base basket bath be bear beat beautiful
because become bed beer before begin behind believe bell below beside best
better between big bird birth bit bite black blood blow blue board boat body
boil bone book boot border born borrow both bottle bottom bowl box boy branch
brave bread break breakfast breath bridge bright bring brother brown brush
build burn bus business busy but butter button buy by
cake call camp can cap capital car card care careful carry case cat catch
cause ceiling cell cent center century certain chair chance change chapter
character charge cheap check cheese chicken child choice choose church circle
city class clean clear clever climb clock close cloth cloud club coal coast
coat coffee cold collect college color come comfort common company compare
complete computer condition connect consider contain continue control cook
cool copy corn corner correct cost cotton cough could count country course
court cover cow crazy cream cross crowd cry cup cut
dance danger dark date daughter day dead deal dear death decide deep degree
deliver depend describe desert design desk destroy detail develop die
difference difficult dig dinner direct dirty discover discuss disease distance
divide do doctor dog dollar door double doubt down draw dream dress drink
drive drop dry during dust duty
each ear early earn earth east easy eat edge education effect effort egg eight
either elect electric else empty end enemy engine enjoy enough enter equal
escape especially even evening event ever every exact example except exchange
excite excuse exercise exist expect expensive experience explain express eye
face fact factory fail fair fall false family famous far farm fashion fast fat
father fault favor fear feed feel female few field fight fill film final find
fine finger finish fire first fish fit five fix flat floor flower fly follow
food foot for force foreign forest forget forgive fork form forward four free
fresh friend from front fruit full fun funny furniture future
game garden gas gate general gentle get gift girl give glad glass go goal god
gold good govern grass gray great green ground group grow guard guess guest
guide gun
hair half hall hand hang happen happy hard hat hate have he head health hear
heart heat heavy help here hide high hill history hit hold hole holiday home
honest hope horse hospital hot hotel hour house how however human hundred
hungry hunt hurry hurt husband
i ice idea if ill imagine important improve in include increase indeed
industry influence inform insect inside instead interest into introduce invent
invite iron island it
job join joke journey joy judge jump just
keep key kick kill kind king kiss kitchen knee knife knock know
lack lady lake land language large last late laugh law lay lazy lead leaf
learn least leave left leg lend length less lesson let letter level library
lie life lift light like line lion lip liquid list listen little live load
local lock long look lose lot loud love low luck lunch
machine mad magazine mail main make male man manage many map mark market marry
mass master match material matter may meal mean measure meat medicine meet
member memory mention metal method middle might milk mind mine minute miss
mistake mix modern moment money month moon more morning most mother mountain
mouth move much music must
name narrow nation natural near necessary neck need needle neighbor neither
nerve net never new news next nice night nine no noise none nor north nose not
note nothing notice now number nurse
obey object ocean of off offer office officer often oil old on once one only
open operate opinion opposite or orange order ordinary organize other ought
out outside over own
page pain paint pair paper parent park part particular party pass past path
patient pattern pay peace pen pencil people per perfect perhaps period person
photograph pick picture piece pig pin pipe place plain plan plant plastic
plate play pleasant please plenty pocket point police polite political pool
poor popular position possible post pot potato pound pour power practice
prepare present president press pretty prevent price print prison private
prize probably problem produce program progress promise proper protect proud
prove provide public pull punish pupil pure purpose push put
quality quarter queen question quick quiet quite
race radio rail rain raise rank rate rather reach read ready real reason
receive recent record red reduce refuse regard region regular relate remain
remember remove rent repair repeat reply report represent require rest result
return rice rich ride right ring rise risk river road rock roll roof room root
rope round row rub rule run
sad safe sail salt same sand save say scale school science score sea search
season seat second secret section see seed seem sell send sense sentence
separate serious serve service set settle seven several sex shade shake shall
shape share sharp she sheep sheet shelf shell shine ship shirt shock shoe
shoot shop short should shoulder shout show shut sick side sight sign silence
silver similar simple since sing single sink sister sit situation six size
skill skin sky sleep slip slow small smell smile smoke snow so soap social
society soft soil soldier solid solve some son song soon sorry sort sound soup
south space speak special speed spell spend spirit spoon sport spread spring
square stage stair stamp stand standard star start state station stay steal
steam steel step stick still stomach stone stop store storm story straight
strange street strength stretch strike strong structure student study stupid
subject succeed such sudden suffer sugar suggest suit summer sun supply
support suppose sure surface surprise sweet swim system
table tail take talk tall taste tax tea teach team tear telephone television
tell temperature ten tend term terrible test than thank that the theater then
there therefore thick thin thing think third thirsty this though thought
thousand thread three throat through throw thus ticket tie tight time tire
title to today together tomorrow tonight too tool tooth top total touch tour
toward towel tower town toy trade traffic train translate travel treat tree
trip trouble truck true trust try turn twelve twenty twice two type
ugly uncle under understand unit unite universe university unless until up
upon use useful usual
valley value various vegetable very victory view village visit voice vote
wait wake walk wall want war warm warn wash waste watch water wave way we weak
wear weather week weight welcome well west wet what wheel when where whether
which while white who whole why wide wife wild will win wind window wine wing
winter wire wise wish with within without woman wonder wood wool word work
world worry worth would wound wrap write wrong
yard year yellow yes yesterday yet you young
""".split())

BAND2 = set("""
accept accident account accurate accuse ache achieve acid active actual adapt
addition adequate adjust admire admit adopt adult advance advantage adventure
advertise advice affair affect afford agency aid aim alarm alcohol alive ally
alter alternative amaze ambition ancient angle ankle announce annoy annual
anxiety apart apologize apparent appeal appetite apply appoint appreciate
approach appropriate approve approximate argue arrange arrest arrow artificial
ash ashamed aside aspect assist associate assume assure athlete atmosphere
atom attach attempt attend attention attitude attract audience author
authority available average avoid awake aware awful awkward
bake balance bar bare bargain bark barrel basic basis battle bay beach beam
bean beard beg behave belong belt bend benefit bet betray beyond bill bind
biology blade blame blank blanket bless blind block boast bold bomb bond bore
boss bother bounce bow brain brake brass breast breathe breed breeze brick
brief brilliant broad budget bull bullet bunch burden bury bush butterfly
cabbage cabin cable calculate calm campaign cancel cancer candidate candle
canvas capable capacity capture carbon career carpet cartoon carve cash cast
castle casual cattle cave cease celebrate ceremony chain chalk challenge
chamber champion channel chaos charm chart chase chat cheat cheek cheer
chemical chest chew chief chimney chip chocolate choke chop cigarette cinema
circuit citizen civil claim clap clash classic clay client cliff climate clinic
clip closet clue clumsy coach code coin collapse colleague column combine
comedy command comment commercial commit committee communicate community
compete complain complex compose compound comprehensive compromise conceal
concentrate concept concern conclude concrete condemn conduct conference
confess confidence confirm conflict confuse congratulate conscious consent
consequence conservative consist constant constitute construct consult consume
contact contemporary content contest context contract contrast contribute
convenient conversation convert convince cooperate cope core corporate
correspond corrupt costume council counter courage crack craft crash crawl
create creature credit creep crew crime crisis critic crop crown crucial crude
cruel crush cube cultivate culture cure curious curl currency current curtain
curve custom cycle
daily dairy dam damage damp dare dash data dawn deaf debate debt decade decay
deceive decent declare decline decorate decrease dedicate defeat defend define
definite delay delicate delight demand democracy demonstrate deny depart
department deposit depress deprive derive descend deserve desire desperate
despite dessert destination detect determine device devil devote diamond diary
dictionary differ digital dignity dimension diminish dip direction disagree
disappear disappoint disaster discipline disclose discount discourage disgust
dismiss disorder display dispute dissolve distinct distinguish distribute
district disturb ditch dive diverse document domain domestic dominate donate
dose dot dozen drag drain drama drift drill drown drug drum drunk due dull
dump dye dynamic
eager eagle earnest ease echo economy edit educate efficient elaborate elbow
elderly element elegant eliminate elsewhere embarrass embrace emerge emergency
emotion emphasis empire employ enable enclose encounter encourage endure
energy enforce engage engineer enhance enormous ensure entertain enthusiasm
entire entitle entry environment episode equip era error essay essential
establish estate estimate eternal ethnic evaluate eventual evidence evil evolve
exaggerate examine exceed excellent exception excess exclude execute exhaust
exhibit expand expense experiment expert expire explode explore export expose
extend extent external extra extreme
fabric facility factor fade faint faith fame fancy fantasy fare farewell
fascinate fatal fate fatigue favorite feast feather feature federal fee fellow
fence fertile festival fiber fiction fierce figure file finance firm flag
flame flash flavor flee fleet flesh flexible flight float flood flour flow
fluid focus fog fold fond forbid forecast forehead formal format former
formula fortune foundation fountain fraction frame framework frank fraud
freeze frequent friction frighten frown fuel fulfill function fund fundamental
funeral fur furious furthermore
gain gallery gamble gang gap garage garbage gather gaze gear gene generate
generous genius genuine gesture giant glance glide glimpse globe glory glove
glow glue goods gorgeous gossip grab grace grade gradual grain grand grant
grape graph grasp grateful grave greet grief grin grind grip groan grocery
gross guarantee guilty gulf gym
habit hail halt hammer handle handsome harbor harm harmony harsh harvest haste
haul hawk hay hazard headache heal heap heaven hedge heel height heir helmet
hence herd hero hesitate highlight hint hire hollow holy honey honor hook
horizon horror host household hug huge humble humor hunger hurricane hut
hydrogen hypothesis
ideal identify identity idle ignore illegal illness illustrate image imitate
immediate immense immigrant impact imply import impose impress incident
incline income indicate individual industrial inevitable infant infect infer
inferior infinite inflation initial initiative injure inner innocent input
inquire insert insight insist inspect inspire install instance instant
institute instruct instrument insult insurance intellect intelligent intend
intense intention interact interfere interior internal international interpret
interrupt interval intervene intimate invest investigate involve isolate issue
item
jail jam jaw jealous jet jewel joint journal jug juice junior jury justice
justify
keen kettle kidney kingdom kit knot knowledge
label labor ladder lag lamb lamp landscape lane lap launch lawn layer leak
lean leap lecture legal legend leisure lemon lens lever liberal liberty
license lid lifetime likely likewise limb limit link liquor literary
literature litter loan lobby locate log logic lonely loop loose lord lorry
loss loyal luggage lump lung luxury
magic magnet magnificent maintain major majority mall mammal manner manual
manufacture manuscript margin marine marvel mask mate mature maximum maybe
meadow meaning meanwhile mechanic medal media medium melt mend mental mercy
mere merge merit merry mess message metaphor migrate mild military mill
mineral minimum minister minor minority miracle mirror miserable mischief
missile mission mist mob mode model moderate modest modify moist mold molecule
monitor monkey monster monument mood moral mortgage mosquito motion motive
motor mount mud multiply murder murmur muscle museum mushroom mutual mystery
myth
nail naked nap narrative nasty native navy neat negative neglect negotiate
nephew nest neutral niece nightmare noble nod nominate nonsense norm normal
notion novel nowhere nuclear nut
oak oath obligation observe obstacle obtain obvious occasion occupy occur odd
odor offend official offset omit ongoing onion operation opponent opportunity
oppose optimistic option oral orbit orchestra organ origin ornament orphan
otherwise outcome outline outlook output outstanding oven overall overcome
overlook overseas overtake owe owl oxygen
pace pack pad palace pale palm pan panel panic parade paragraph parallel
parcel pardon participate partner passage passenger passion passive pasture
patch patience patrol pause pave peak peasant peculiar peel peer penalty
penetrate pension perceive percent perform permanent permit persist personal
personality perspective persuade pest petrol phase phenomenon philosophy
phrase physical pile pill pillar pillow pilot pinch pine pioneer pit pity
plague platform plead pleasure pledge plot plough plug plunge poem poet poison
pole policy polish pollute pond port portion portrait pose possess postpone
potential pottery poverty powder practical praise pray precious precise
predict prefer pregnant prejudice preliminary premise prescribe presence
preserve pressure prestige presume pretend previous prey priest primary prime
primitive principal principle prior priority privilege procedure proceed
process proclaim profession professor profile profit profound prohibit project
prominent promote prompt pronounce proof propaganda proportion proposal
propose prospect prosper protest prototype province provision provoke
psychology publish pump punch purchase pursue puzzle
qualify quantity quarrel quest queue quit quote
rabbit rack radiate radical rag rage raid rainbow random range rapid rare rat
ratio ration rational raw ray razor react realize realm rear reasonable rebel
recall receipt recipe recognize recommend recover recruit recycle refer
reflect reform refresh refuge regarding regime register regret regulate reign
reject rejoice relative relax release relevant reliable relief religion
reluctant rely remark remedy remind remote renew replace reputation request
rescue research resemble reserve reside resign resist resolve resort resource
respect respond responsible restore restrain restrict retain retire retreat
reveal revenge revenue reverse revise revive revolution reward rhythm rib
ribbon rid ridge ridiculous rifle rigid riot ripe ritual rival roar roast rob
robot rocket rod romance rot rotate rough route routine royal rubber rubbish
rude ruin rumor rural rush rust
sack sacred sacrifice saddle sake salary sample sanction sanity satellite
satisfy sauce sausage scan scandal scar scarce scare scatter scene schedule
scheme scholar scope scratch scream screen screw script sculpture seal secure
seek segment seize seldom select senate senior sensitive sequence sergeant
session severe sew shadow shaft shallow shame shave shed shelter shield shift
shiver shore shrink shrug sibling siege sigh signal significant silk silly sin
sincere sip site skeleton sketch ski skip skirt skull slam slap slave slice
slide slight slim slope slot smart smash smooth snake snap sneak soak soar
sober socket sole solemn solution somewhat soul source souvenir sow span spare
spark sparkle specialist species specific specimen spectacle spectrum
speculate sphere spice spider spill spin spine spiral spit splash split spoil
sponsor spot spouse spray sprinkle spy squad squeeze stab stable stack staff
stain stake stall stance staple stare starve statement statistic statue status
steady steep steer stem stereotype stern stiff stimulate sting stir stock
stool stove strain strand strap strategy straw stream strengthen stress strict
stride strip stripe strive stroke stroll struggle stubborn stuff stumble stun
submit subsequent substance substitute subtle suburb subway sufficient suicide
suitable sum summary summit superior supermarket supervise supplement supreme
surgery surplus surrender surround survey survive suspect suspend suspicion
sustain swallow swamp swap swear sweat sweep swell swift swing switch sword
symbol sympathy symptom syndrome synthesis
tackle tag talent tank tap tape target task tease technical technique
technology telescope temper temple temporary tempt tenant tender tension tent
terminal terrify territory terror text texture theme theory therapy thesis
thorough threat thrill thrive throne thumb thunder tide tidy timber timid tin
tiny tip tissue toast tobacco toe tolerate tomb ton tone tongue torture toss
tough tournament trace track tractor tradition tragedy trail trait transfer
transform transit transmit transport trap trash tray tremble tremendous trend
trial triangle tribe trick trigger trim triumph troop tropical trunk tube tuck
tug tumble tune tunnel twin twist typical
ultimate umbrella unable uncertain uncomfortable undergo undertake undo uneasy
unemployment unfair uniform union unique unlike unusual upset upstairs urban
urge urgent usage utility utter
vacant vacation vague vain valid van vanish vapor variety vary vast vehicle
veil vein venture verse version vertical vessel veteran via vibrate vice
vicious victim video vigorous villain vine violate violent violin virtual
virtue virus visible vision visual vital vitamin vivid vocabulary volcano
volume voluntary volunteer vow voyage vulnerable
wag wage wagon waist wander ward wardrobe warehouse warrant warrior wary wax
weapon weave web wedding weed weekend weep weird welfare whale wheat whereas
whip whisper whistle wicked widespread widow width wilderness willing wisdom
withdraw witness wonderful worship worthwhile wrist
yawn yield youth zeal zone
""".split())

BANDS = ("K1", "K2", "off-list")


# ---------------------------------------------------------------------------
# Irregular forms. Rules cannot reach these, and they are the commonest words
# in the language, so getting them wrong is not a rounding error.
# ---------------------------------------------------------------------------
IRREGULAR = {
    # be / have / do
    "was": "be", "were": "be", "is": "be", "are": "be", "am": "be",
    "been": "be", "being": "be", "had": "have", "has": "have",
    "having": "have", "did": "do", "does": "do", "done": "do", "doing": "do",
    # the rest, roughly by frequency
    "went": "go", "gone": "go", "goes": "go", "said": "say", "says": "say",
    "made": "make", "took": "take", "taken": "take", "came": "come",
    "saw": "see", "seen": "see", "got": "get", "gotten": "get",
    "knew": "know", "known": "know", "thought": "think", "found": "find",
    "gave": "give", "given": "give", "told": "tell", "became": "become",
    "left": "leave", "felt": "feel", "brought": "bring", "began": "begin",
    "begun": "begin", "kept": "keep", "held": "hold", "wrote": "write",
    "written": "write", "stood": "stand", "heard": "hear", "meant": "mean",
    "met": "meet", "ran": "run", "paid": "pay", "sat": "sit",
    "spoke": "speak", "spoken": "speak", "led": "lead", "grew": "grow",
    "grown": "grow", "lost": "lose", "fell": "fall", "fallen": "fall",
    "sent": "send", "built": "build", "understood": "understand",
    "drew": "draw", "drawn": "draw", "broke": "break", "broken": "break",
    "spent": "spend", "rose": "rise", "risen": "rise", "drove": "drive",
    "driven": "drive", "bought": "buy", "wore": "wear", "worn": "wear",
    "chose": "choose", "chosen": "choose", "ate": "eat", "eaten": "eat",
    "sold": "sell", "won": "win", "taught": "teach", "caught": "catch",
    "threw": "throw", "thrown": "throw", "flew": "fly", "flown": "fly",
    "forgot": "forget", "forgotten": "forget", "slept": "sleep",
    "swam": "swim", "swum": "swim", "drank": "drink", "drunk": "drink",
    "sang": "sing", "sung": "sing", "rode": "ride", "ridden": "ride",
    "dealt": "deal", "sought": "seek", "fought": "fight", "hung": "hang",
    "shot": "shoot", "stuck": "stick", "struck": "strike", "woke": "wake",
    "woken": "wake", "hid": "hide", "hidden": "hide", "laid": "lay",
    "lain": "lie", "arose": "arise", "arisen": "arise", "bore": "bear",
    "borne": "bear", "bit": "bite", "bitten": "bite", "blew": "blow",
    "blown": "blow", "dug": "dig", "fed": "feed", "fled": "flee",
    "froze": "freeze", "frozen": "freeze", "knelt": "kneel",
    "lent": "lend", "lit": "light", "shone": "shine", "shrank": "shrink",
    "slid": "slide", "sprang": "spring", "stole": "steal", "stolen": "steal",
    "swept": "sweep", "tore": "tear", "torn": "tear", "wove": "weave",
    "woven": "weave", "withdrew": "withdraw", "withdrawn": "withdraw",
    "swore": "swear", "sworn": "swear", "wound": "wind",
    # nouns
    "men": "man", "women": "woman", "children": "child", "people": "person",
    "feet": "foot", "teeth": "tooth", "geese": "goose", "mice": "mouse",
    "lives": "life", "wives": "wife", "knives": "knife", "leaves": "leaf",
    "halves": "half", "shelves": "shelf", "wolves": "wolf",
    "thieves": "thief", "selves": "self", "loaves": "loaf",
    "calves": "calf", "scarves": "scarf", "criteria": "criterion",
    "phenomena": "phenomenon", "analyses": "analysis", "crises": "crisis",
    "theses": "thesis", "indices": "index", "matrices": "matrix",
    # adjectives / adverbs
    "better": "good", "best": "good", "worse": "bad", "worst": "bad",
    "more": "much", "most": "much", "less": "little", "least": "little",
    "further": "far", "furthest": "far", "farther": "far", "farthest": "far",
    "elder": "old", "eldest": "old",
    # -ie- verbs: the rules would reach "dye" from "dying"
    "dying": "die", "lying": "lie", "tying": "tie",
}

# Clitics the tokeniser keeps attached, because "don't" is one thing you either
# say or don't. For counting they belong with their head word.
CLITIC = (("n't", None), ("'ll", "will"), ("'re", "be"), ("'ve", "have"),
          ("'m", "be"), ("'d", "would"), ("'s", None))

# The only two-letter bases a strip is allowed to land on. See `Lexicon._pick`.
SHORT_STEMS = frozenset(("go", "be", "do"))

# Words that end in an inflectional suffix and are not inflected. Without this
# list "news" folds to "new", "business" to "busy", and "during" to "dure".
NOT_INFLECTED = set("""
this his has was is as us yes gas bus plus thus news series species means
glass class grass pass mass press dress cross loss boss guess process business
across always perhaps sometimes towards besides otherwise unless
during evening morning something anything nothing everything
bring sing ring king thing wing spring string ceiling feeling
does goes toes woes shoes
odds needs goods thanks clothes stairs jeans pants shorts glasses scissors
ours yours theirs hers its
red bed bad dead need seed feed speed indeed instead ahead
deed reed weed heed breed greed creed bleed
world word ward sword lord
""".split())

# Pairs where the derivation rules find a real word that is not the root.
# Only entries that actually occur are worth carrying; the length and
# frequency guards in `_derive` catch the rest.
NOT_DERIVED = set("""
corner dinner butter water matter letter number summer winter finger offer
order other mother father brother weather rather
under over very every many any only company family
morning evening during person reason season carry marry hurry
level novel travel lovely inside outside interest present
early forest topic question several disease display understand income
insect increase shower listen little middle people
final finance finish mistake mister master manner matter study
""".split())


def _wordlists():
    """Two lists, because the two folds need different levels of trust.

    `known` is permissive — anything either dictionary has heard of. Inflection
    is validated against it, and that is safe: "-ed" only strips if the result
    is a word, and an exact suffix relationship rarely lands on a wrong one.

    `roots` is strict, because derivation is where a permissive list does real
    damage. cmudict is full of surnames (`libert`, `cooke`, `earlie`) and
    /usr/share/dict/words is full of archaic entries (`vocabular`, `sente`);
    each would happily absorb a word into a root nobody has ever said. Their
    *intersection* is neither — about 36k words that are both pronounceable
    English and dictionary English. That, plus the frequency bands, is what
    a derivation is allowed to fold onto.
    """
    ok = re.compile(r"[a-z']+")
    cmu = set()
    try:
        import cmudict
        cmu = {w.lower() for w in cmudict.dict() if ok.fullmatch(w.lower())}
    except Exception:
        pass
    web = set()
    p = "/usr/share/dict/words"
    if os.path.exists(p):
        try:
            with open(p, encoding="utf-8", errors="ignore") as f:
                web = {w for w in (l.strip().lower() for l in f)
                       if w and ok.fullmatch(w)}
        except OSError:
            pass
    bands = set(BAND1) | set(BAND2)
    known = bands | cmu | web
    roots = bands | ((cmu & web) if (cmu and web) else (cmu or web))
    return known, roots


class Lexicon:
    """Fold word forms to lemmas and to word families. Cheap after the first call."""

    # Derivational suffixes, longest first: each is a suffix and the endings to
    # try in its place ("happiness" -> happi + {"", "ness"} -> "happy" via -y).
    SUFFIX = [
        ("ically", ["", "ic", "y"]), ("ization", ["e", "", "ize"]),
        ("isation", ["e", "", "ise"]), ("ability", ["", "able", "e"]),
        ("ibility", ["ible", "e"]), ("fulness", ["", "ful"]),
        ("lessness", ["", "less"]), ("iness", ["y"]), ("ness", [""]),
        ("ation", ["e", "", "ate"]), ("ition", ["e", "ite"]),
        ("ional", ["ion", "e", ""]), ("tion", ["te", "t", ""]),
        ("sion", ["de", "d", "se", ""]), ("ement", ["e", ""]), ("ment", [""]),
        ("ously", ["ous"]), ("ily", ["y"]), ("ly", [""]),
        ("ical", ["ic", "e", ""]), ("ative", ["ate", "e", ""]),
        ("ive", ["e", ""]), ("ance", ["e", ""]), ("ence", ["e", ""]), ("ent", ["", "e"]),
        ("ancy", ["ant", "e"]), ("ency", ["ent", "e"]), ("ity", ["e", ""]),
        ("ify", ["y", ""]), ("ize", ["e", ""]), ("ise", ["e", ""]),
        ("able", ["e", "", "ate"]), ("ible", ["e", ""]),
        ("ful", ["", "e"]), ("less", ["", "e"]), ("ish", ["", "e"]),
        ("ous", ["e", "", "y"]), ("ist", ["e", "", "y"]), ("ism", ["e", ""]),
        ("ship", [""]), ("hood", [""]), ("dom", [""]),
        ("ier", ["y"]), ("iest", ["y"]), ("er", ["", "e"]), ("or", ["", "e"]),
        ("est", ["", "e"]), ("al", ["", "e"]), ("ic", ["", "e", "y"]),
        ("y", ["", "e"]),
    ]
    # Only prefixes whose meaning is transparent and whose base stands alone.
    # `in-`/`re-` are deliberately absent: "increase" is not a kind of crease
    # and "report" is not a kind of port, and no rule can tell those from
    # "inactive" and "rewrite".
    PREFIX = ["counter", "under", "over", "semi", "anti", "auto", "multi",
              "non", "mis", "dis", "un"]

    def __init__(self, known=None, roots=None):
        if known is None or roots is None:
            k, r = _wordlists()
            known = known if known is not None else k
            roots = roots if roots is not None else r
        self.known, self.roots = known, roots
        self._lemma = {}
        self._family = {}

    def is_word(self, w):
        return w in self.known

    # -------------------------------------------------------------- inflection
    def lemma(self, w):
        """Fold inflections: plurals, tenses, participles, comparatives."""
        if w not in self._lemma:
            self._lemma[w] = self._inflect(w)
        return self._lemma[w]

    def _inflect(self, w):
        w = w.lower().strip("'")
        for suf, standalone in CLITIC:
            if w.endswith(suf) and len(w) > len(suf):
                head = w[:-len(suf)]
                return self.lemma(head)
            if w == suf and standalone:
                return standalone
        if w in IRREGULAR:
            return IRREGULAR[w]
        if w in NOT_INFLECTED or len(w) < 4:
            return w
        # -ies / -ied: studies, studied -> study
        if len(w) > 4 and w[-3:] in ("ies", "ied") and self.is_word(w[:-3] + "y"):
            return w[:-3] + "y"
        # -es after a sibilant or -o: boxes, watches, wishes, goes
        if w.endswith("es") and len(w) > 4:
            b = w[:-2]
            if (b[-1:] in ("s", "x", "z", "o") or b[-2:] in ("ch", "sh")) \
                    and self.is_word(b):
                return b
        # plain -s
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            b = self._pick([w[:-1]])
            if b:
                return b
        # -ed, incl. dropped -e (liked) and doubled consonant (stopped)
        if w.endswith("ed") and len(w) > 3:
            cands = [w[:-2]]
            if len(w) > 5 and w[-3] == w[-4]:
                cands.append(w[:-3])
            cands.append(w[:-1])
            b = self._pick(cands)
            if b:
                return b
        # -ing, same two spelling changes (making, running). The threshold is
        # 5 and not 6 so that "going" and "using" reach their stems; the
        # three-letter words that end in -ing ("bring", "thing") are held back
        # by NOT_INFLECTED and by the stem checks in `_pick`.
        if w.endswith("ing") and len(w) > 4:
            stem = w[:-3]
            cands = [stem, stem + "e"]
            if len(stem) > 2 and stem[-1] == stem[-2]:
                cands.insert(0, stem[:-1])
            b = self._pick(cands)
            if b:
                return b
        return w

    def _pick(self, cands):
        """Choose a base from several spellings of the same strip.

        Two passes, and the order is the whole point. "called" can be read as
        call+ed or calle+d, and "making" as make-e+ing or mak+ing; cmudict has
        `calle` and `mak` (both surnames), so a single pass over a permissive
        list picks the wrong one. Anything in the strict root list wins first;
        only if nothing does do we fall back to the permissive list, and then
        only for words long enough not to be a stray name — which is how
        "laptops" still reaches "laptop", a word no 1934 dictionary contains.
        """
        for b in cands:
            # A two-letter base is almost always the wrong read: "used" strips
            # to "us" one letter before it strips to "use", and "using" to "us"
            # before "use". English has only three two-letter verb stems, so
            # name them rather than banning the length outright — otherwise
            # "going" and "doing" lose their roots to fix "used".
            if (len(b) >= 3 or b in SHORT_STEMS) and b in self.roots:
                return b
        for b in cands:
            if len(b) >= 4 and self.is_word(b):
                return b
        return None

    # -------------------------------------------------------------- derivation
    def family(self, w):
        """Fold derivations too, so care/careful/carefully/careless are one."""
        if w not in self._family:
            self._family[w] = self._derive(self.lemma(w), 0)
        return self._family[w]

    def _plausible(self, b):
        """A candidate base must look like a word people actually use.

        `roots` already excludes surnames and archaisms. The length rule
        catches what is left: even a clean list contains "fin" and "sugg", and
        a three- or four-letter target is the one most likely to be hit by
        accident. Common words are exempt — "play" and "care" are short and
        are exactly the roots we want to fold onto.

        NOT_DERIVED is deliberately not consulted here: it says "do not take
        this word apart", not "this word may not be a root". Checking it here
        would stop "earlier" folding onto "early" and send it somewhere worse.
        """
        if b not in self.roots:
            return False
        return b in BAND1 or b in BAND2 or len(b) >= 5

    def _demotes(self, w, b):
        """Would folding `w` onto `b` move it off the frequency lists?

        The published bands are counted in families, so a word that appears on
        one is already a family head. "especially" is K1; strip -ly and you get
        "especial", a real but rare word, and a common adverb silently becomes
        evidence of a wide vocabulary. When the root is listed too ("careful"
        -> "care") the fold is right and this returns False.
        """
        listed = w in BAND1 or w in BAND2
        return listed and not (b in BAND1 or b in BAND2)

    def _derive(self, w, depth):
        if depth > 3 or len(w) < 5 or w in NOT_DERIVED:
            return w
        for suf, reps in self.SUFFIX:
            if not w.endswith(suf):
                continue
            stem = w[:-len(suf)]
            if len(stem) < 3:
                continue
            cands = [stem + r for r in reps]
            if stem[-1] == stem[-2]:                     # runner -> run
                cands.append(stem[:-1])
            for b in cands:
                if b != w and self._plausible(b) and not self._demotes(w, b):
                    return self._derive(self.lemma(b), depth + 1)
        for pre in self.PREFIX:
            if w.startswith(pre) and len(w) - len(pre) >= 4:
                b = w[len(pre):]
                if self._plausible(b) and not self._demotes(w, b):
                    return self._derive(self.lemma(b), depth + 1)
        return w

    # -------------------------------------------------------------------- band
    def band(self, w):
        """Which thousand of English a word belongs to. Folded to family first,
        because that is the unit the published lists are counted in.

        `unknown` is not a fourth band, it is the escape hatch: a transcript
        carries proper nouns (Azure, Kimi) and whatever the recogniser made of
        a mumble. Counting those as rare vocabulary would flatter the number
        that this whole report exists to keep honest.
        """
        f = self.family(w)
        if f in BAND1:
            return "K1"
        if f in BAND2:
            return "K2"
        # a one- or two-letter "word" is the recogniser mishearing something
        return "off-list" if (f in self.roots and len(f) > 2) else "unknown"


_SHARED = None


def shared():
    """One Lexicon per process — building the known-word set costs ~0.4s."""
    global _SHARED
    if _SHARED is None:
        _SHARED = Lexicon()
    return _SHARED


# ---------------------------------------------------------------------------
# Estimating the vocabulary behind the sample
# ---------------------------------------------------------------------------
def chao1(counts):
    """How many families you'd have used, given unlimited talking.

    Every corpus undercounts its speaker: you cannot observe a word they did
    not happen to need. The rate of undercounting is readable from the sample
    itself, though — a speaker still producing lots of once-only words has
    plenty more unsaid, and one who repeats everything has nearly run out.

        S_est = S_obs + f1**2 / (2 * f2)

    with f1 the number used exactly once and f2 exactly twice (Chao 1984).
    It is a *lower* bound on the true size, and it says nothing about words
    that are understood but never spoken, which is most of them.
    """
    f1 = sum(1 for n in counts.values() if n == 1)
    f2 = sum(1 for n in counts.values() if n == 2)
    obs = len(counts)
    if not f2:
        return obs + f1 * (f1 - 1) / 2.0 if f1 > 1 else float(obs)
    return obs + (f1 * f1) / (2.0 * f2)


def profile(word_counts, lx=None):
    """The whole picture, from a Counter of raw word forms.

    Returns per-unit counts (forms, lemmas, families), the K1/K2/off-list
    split of both families and running words, and the Chao1 estimate.
    """
    from collections import Counter
    lx = lx or shared()
    lemmas, families = Counter(), Counter()
    members = {}
    for w, n in word_counts.items():
        lemmas[lx.lemma(w)] += n
        f = lx.family(w)
        families[f] += n
        members.setdefault(f, set()).add(w)

    fam_band, tok_band, by_band = Counter(), Counter(), {}
    for f, n in families.items():
        b = lx.band(f)
        fam_band[b] += 1
        tok_band[b] += n
        by_band.setdefault(b, []).append(f)

    # The headline counts exclude names and recogniser noise: they are not
    # vocabulary, and they are the easiest way to accidentally inflate a
    # "words I know" figure by several percent.
    real = Counter({f: n for f, n in families.items() if lx.band(f) != "unknown"})
    return {
        "tokens": sum(word_counts.values()),
        "forms": len(word_counts),
        "lemmas": len(lemmas),
        "families": len(real),
        "families_all": len(families),
        "unknown": fam_band.get("unknown", 0),
        "lemma_counts": lemmas,
        "family_counts": families,
        "family_members": members,
        "by_band": by_band,
        "fam_band": fam_band,
        "tok_band": tok_band,
        "chao1": chao1(real),
        "hapax_families": sum(1 for n in real.values() if n == 1),
    }
