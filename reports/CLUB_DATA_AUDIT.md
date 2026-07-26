# Club football — data-source audit

**Retrieved 2026-07-26T20:32:26Z.** Every claim below was verified by
downloading the file and inspecting it, not from documentation or memory.
Nothing here has been ingested into the pipeline yet; this audit exists to
decide whether a club expansion is viable and on what terms.

Reproduce with the probe commands quoted under each source.

---

## 1. football-data.co.uk — main European league files

| field | value |
|---|---|
| **URL pattern** | `https://www.football-data.co.uk/mmz4281/{season}/{div}.csv` (e.g. `.../2526/E0.csv`) |
| **Verified divisions** | `E0` (Premier League), `D1` (Bundesliga), `SP1` (La Liga), `I1` (Serie A), `F1` (Ligue 1) — all HTTP 200 for 2024-25 and 2025-26 |
| **Coverage** | 1993-94 → 2025-26. The 2025-26 season is **complete** (380 matches, 2025-08-15 → 2026-05-24) |
| **Most recent match** | 2026-05-24 (season over; 2026-27 files not yet published) |
| **Rows/season** | 380 (20 teams); 462 in 1993-94 (22 teams) |
| **Extra-time / penalties** | **Not applicable** — league play only, no knockout ties. No ET or shootout columns exist |
| **Post-match information** | Match statistics (shots, cards) are post-match by nature and must only ever be used as *lagged* features for later matches, never for the match they describe |

### Fields, by era (verified per season)

| season | n | shots | shots on target | half-time | kickoff time | closing odds |
|---|---|---|---|---|---|---|
| 1993-94 | 462 | ✗ | ✗ | ✗ | ✗ | ✗ |
| 1999-00 | 380 | ✗ | ✗ | ✓ | ✗ | ✗ |
| 2000-01 | 380 | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2010-11 | 380 | ✓ | ✓ | ✓ | ✗ | ✗ |
| 2015-16 | 380 | ✓ | ✓ | ✓ | ✗ | Pinnacle only |
| 2019-20 | 380 | ✓ | ✓ | ✓ | ✓ | ✓ full |
| 2025-26 | 380 | ✓ | ✓ | ✓ | ✓ | ✓ full |

Three era boundaries that constrain the model:

* **Shots and corners begin 2000-01.** Before that, goals only.
* **Kickoff times begin 2019-20.** Earlier seasons need the existing
  conservative whole-day ordering rule; from 2019-20 the intra-day ordering
  the international pipeline never had becomes possible.
* **Closing odds begin ~2015-16** (Pinnacle) and become full/consensus in
  2019-20. This bounds the market-baseline sample: roughly 4,200 EPL matches
  since 2015-16, or ~21,000 across the big five.

Core fields in 2025-26: `Date, Time, HomeTeam, AwayTeam, FTHG, FTAG, FTR,
HTHG, HTAG, HTR, Referee, HS, AS, HST, AST, HC, AC, HF, AF, HY, AY, HR, AR`
plus 69 odds columns.

**Missing-data rate on core fields, 2025-26 EPL: 0.0%** (every core column
fully populated across all 380 rows).

### Licensing — usable, but do not redistribute

The site describes itself as "a free football betting portal" and says to
"Simply download for free the available files". The footer reads
"© Football-Data. Liability Disclaimer. All Rights Reserved."
`notes.txt` (the field-definition document) contains **no** licensing,
redistribution, or attribution statement at all.

So: downloading and analysing is plainly the intended use; **redistribution
is not granted**. This fits the pattern the repo already uses for the
international sources — download at runtime, cache under `data/`, keep
`data/` gitignored, never commit the raw files. No change of practice needed.

### Known limitations (verified, not assumed)

1. **Ragged CSVs silently lose matches.** `2004-05/E0.csv` has 45 rows with
   62 fields against a 57-field header. `pandas.read_csv` raises; the common
   workarounds (`on_bad_lines='skip'`, `engine='python'`) **silently drop
   those 45 rows** — 12% of the season, all at the end. Verified fix: read
   with `csv.reader`, pad the header to the widest row, then drop all-empty
   rows → recovers all 380. Any ingester must assert the expected row count
   per season rather than trusting the parser.
2. **Trailing empty rows.** 1993-94 reports 552 rows, of which 90 are
   entirely blank (real count 462); 1999-00 reports 552 with 172 blank (real
   count 380). `dropna(how='all')` handles it, but a naive row count is wrong.
3. **UTF-8 BOM on the first header field** — reads as `ï»¿Div` under
   latin-1. Use `utf-8-sig` or strip it.
4. **Post-2020 home advantage break.** Not measured here, but the 2020-21
   empty-stadium season is a known structural break that any home-advantage
   parameter must accommodate.

---

## 2. football-data.co.uk — MLS ("new leagues" file)

| field | value |
|---|---|
| **URL** | `https://www.football-data.co.uk/new/USA.csv` |
| **Coverage** | 6,054 matches, 2012-03-10 → **2026-07-23** (three days before retrieval — this source is live) |
| **Seasons** | 2012 → 2026 (calendar-year seasons; 2026 in progress, 238 matches so far) |
| **Contents** | MLS only — one league per country file |
| **Fields** | `Country, League, Season, Date, Time, Home, Away, HG, AG, Res` + 15 closing-odds columns (Pinnacle, Bet365, Max, Avg, BFE) |
| **Missing** | **No shots, corners, cards, half-time scores, or referee.** Goals and odds only |

### Shootouts — better than expected

MLS playoff ties are decided by penalty shootout. Verified: across all 6,054
rows there are **zero** rows where `Res` disagrees with `HG`/`AG`, and the
draw rate in the November–December playoff window (25.9%, n=228) matches the
regular season (25.0%). The file therefore records the **90-minute score**
and marks shootout-decided playoff games as draws — exactly the convention
this project already uses. Penalty goals are never folded into `HG`/`AG`.

The cost: there is **no stage column and no shootout flag**, so a
playoff shootout is indistinguishable from an ordinary draw without joining
an external fixture list. That blocks MLS *advancement* modelling (who went
through) but not the core 90-minute model.

### Other MLS-specific notes

Home win rate is 49.5% (H 2998 / D 1517 / A 1539) — materially higher than
typical European leagues, consistent with MLS travel distances. Home
advantage must be fitted per competition, never shared with Europe.

---

## 3. ClubElo — cross-league strength prior and benchmark

| field | value |
|---|---|
| **URL** | `http://api.clubelo.com/{YYYY-MM-DD}` (all clubs on a date) and `http://api.clubelo.com/{Club}` (full history) |
| **Verified** | Both endpoints HTTP 200. Snapshot for 2026-07-25 returned 32 KB; `Arsenal` returned 6,507 rating rows, 1946-07-07 → 2026-12-31 |
| **Fields** | `Rank, Club, Country, Level, Elo, From, To` — a step function of rating validity intervals, so the rating *before* any given date is directly recoverable (leakage-safe by construction) |
| **Coverage** | ~40 European countries, levels 1–2, plus UCL/UEL/ECL groupings |
| **Extra time / penalties** | Not applicable — ratings, not match records |

### Licensing — UNRESOLVED, treat as blocking for any dependency

I could not retrieve any terms-of-use, licensing, or attribution statement
from `clubelo.com/API` or `clubelo.com/About`; the pages returned navigation
markup only, with no licensing text present in the fetched HTML. The
commonly-repeated claim that the API is "free for non-commercial use" is
**not something I could verify**, and I am not willing to assert it.

**Recommendation:** do not build a required dependency on ClubElo until its
terms are confirmed by a human reading the site directly. It is valuable but
strictly optional — a benchmark and a cross-league prior, both of which the
system can live without.

---

## 4. Sources deliberately NOT used

| source | reason |
|---|---|
| **Understat** | The obvious free xG source for the big five, but its terms make automated collection questionable. CLAUDE.md forbids data collected in violation of a site's terms. Not used pending an explicit terms check |
| **FBref / Sports-Reference** | Same reasoning; the site publishes explicit guidance discouraging automated scraping |
| **API-Football, other freemium APIs** | Free tiers exist but are rate-capped and account-gated; conflicts with the "works with no API key" requirement |

**Mitigation:** shots on target from football-data.co.uk (available from
2000-01, 0% missing in recent seasons) captures a large share of what xG
would provide, with no terms risk. That is the recommended substitute.

---

## Bottom line

The data supports a club expansion. Specifically:

* **Big-five European leagues**, 1993-94 → 2025-26, goals throughout, shots
  from 2000-01, ~21,000 matches with closing odds since 2015-16.
* **MLS**, 2012 → live (2026-07-23), goals and odds only, correct
  90-minute/shootout convention already.
* **A real market baseline is available for the first time** — the single
  biggest scientific upgrade over the international model, which never had one.

The two real costs, both confirmed above rather than speculated:

1. **Ingestion is not "just read_csv".** Three distinct malformations
   (ragged rows, blank padding, BOM) each corrupt a naive read, and one of
   them fails *silently*. Row-count assertions per season are mandatory.
2. **Entity resolution.** One season of the big five alone contains 96
   distinct club names, in a terse house style (`Ath Bilbao`, `Ath Madrid`,
   `Man United`, `Inter`, `Betis`, `Sociedad`) that will not join to ClubElo
   or any other source without a hand-curated alias table. Across 26 seasons
   with promotion and relegation this is the dominant time cost — as
   predicted, the modelling is not the hard part.
