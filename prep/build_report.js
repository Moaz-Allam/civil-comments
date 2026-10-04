/*
 * build_report.js -- writes the Phase 2 report (ACM two-column layout, same
 * page setup as the Phase 1 report) from the JSON files in prep/out and the
 * figures in prep/figures. Every number in the text is read from those files.
 *
 * Run:  node prep/build_report.js        (needs the "docx" npm package)
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell, ExternalHyperlink,
  WidthType, AlignmentType, SectionType, BorderStyle, ShadingType,
} = require("docx");

const OUT = path.join(__dirname, "out");
const FIG = path.join(__dirname, "figures");
const DEST = path.join(__dirname, "..", "Phase2_Data_Preparation_Report.docx");
const J = (n) => JSON.parse(fs.readFileSync(path.join(OUT, n), "utf8"));
const V = J("01_load.json"), A = J("02_analysis.json"), C = J("03_cleaning.json");
const W2 = J("04_word2vec.json"), K = J("05_correlations.json");
const DRIVE_URL = "https://drive.google.com/drive/folders/1zVfZamzpx0P-tyvNQ8fbEE2nyBqw1Dcy?usp=sharing";   // link to the cleaned dataset on Google Drive

// ------------------------------------------------------------------ number formatting
const N = (x) => Math.round(Number(x)).toLocaleString("en-US");
const F = (x, d = 2) => Number(x).toFixed(d);
const P = (frac, d = 2) => (100 * Number(frac)).toFixed(d) + "%";   // fraction -> percent
const Pp = (pct, d = 2) => Number(pct).toFixed(d) + "%";            // already a percent
const sgn = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const MB = (b) => (b / 1e6).toFixed(0) + " MB";
const day = (s) => s.slice(0, 10);
const ord = (n) => { const v = n % 100; const sfx = (v >= 11 && v <= 13) ? "th" : ({ 1: "st", 2: "nd", 3: "rd" }[n % 10] || "th"); return N(n) + sfx; };

// ------------------------------------------------------------------ text helpers
const BODY = 20, SMALL = 16, TINY = 15;
const MONO = "Consolas";
function runs(str, base = {}) {
  const parts = String(str).split(/(`[^`]+`|\*\*[^*]+\*\*|_\{[^}]+\}_)/g).filter((s) => s.length);
  return parts.map((s) => {
    if (s.startsWith("`")) return new TextRun({ ...base, text: s.slice(1, -1), font: MONO, size: (base.size || BODY) - 2 });
    if (s.startsWith("**")) return new TextRun({ ...base, text: s.slice(2, -2), bold: true });
    if (s.startsWith("_{")) return new TextRun({ ...base, text: s.slice(2, -2), italics: true });
    return new TextRun({ ...base, text: s });
  });
}
let firstAfterHeading = true;
const body = (str) => {
  const p = new Paragraph({
    children: runs(str), alignment: AlignmentType.JUSTIFIED, spacing: { line: 211 },
    indent: firstAfterHeading ? undefined : { firstLine: 180 },
  });
  firstAfterHeading = false;
  return p;
};
const bullet = (str) => new Paragraph({
  children: runs(str), alignment: AlignmentType.JUSTIFIED, spacing: { line: 211, after: 20 },
  indent: { left: 240, hanging: 160 },
});
function h1(num, title) {
  firstAfterHeading = true;
  return new Paragraph({ children: [new TextRun({ text: `${num}  ${title.toUpperCase()}`, bold: true, size: 22 })],
    spacing: { before: 180, after: 60 }, keepNext: true });
}
function h2(num, title) {
  firstAfterHeading = true;
  return new Paragraph({ children: [new TextRun({ text: `${num}  ${title}`, bold: true, italics: true })],
    spacing: { before: 120, after: 45 }, keepNext: true });
}
function plainHead(text) {
  firstAfterHeading = true;
  return new Paragraph({ children: [new TextRun({ text, bold: true })], spacing: { before: 140, after: 45 }, keepNext: true });
}

// ------------------------------------------------------------------ figures
let figNo = 0;
const figRef = {};
function figure(key, file, caption, widthIn = 3.25) {
  figNo += 1;
  if (figRef[key] !== figNo) throw new Error(`figure "${key}" appears as ${figNo} but FIGS says ${figRef[key]}`);
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const px = widthIn * 96;
  firstAfterHeading = true;
  return [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: px, height: px * h / w } })] }),
    new Paragraph({ spacing: { before: 55, after: 120 },
      children: runs(`**Figure ${figNo}:** ${caption}`, { size: SMALL }) }),
  ];
}
// Figures are numbered in order of appearance; the text refers to them by key.
// Keys are pre-assigned so prose written before a figure can cite it.
const FIGS = ["labeldist", "labels", "annot", "length", "stats", "pub", "dist", "xcorr", "allcorr", "ycorr"];
FIGS.forEach((k, i) => (figRef[k] = i + 1));
const Fg = (k) => `Figure ${figRef[k]}`;

// ------------------------------------------------------------------ tables
let tabNo = 0;
const COL = 4802, FULL = 10080;
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = (sz = 4, color = "000000") => ({ style: BorderStyle.SINGLE, size: sz, color });
function table(caption, widths, header, rows, aligns) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (txt, i, isHead, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead ? rule() : none, bottom: isHead ? rule() : isLast ? rule() : rule(2, "BFBFBF"),
      left: none, right: none },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "EFEFEF" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ spacing: { line: 195 },
      alignment: (aligns && aligns[i] === "r") ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(txt, { size: TINY, bold: isHead || undefined }) })],
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => mk(h, i, true, false)) })];
  rows.forEach((r, k) => trs.push(new TableRow({ cantSplit: true,
    children: r.map((c, i) => mk(c, i, false, k === rows.length - 1)) })));
  firstAfterHeading = true;
  return [
    new Paragraph({ spacing: { before: 140, after: 60 }, keepNext: true,
      children: runs(`**Table ${tabNo}:** ${caption}`, { size: SMALL }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, rows: trs }),
    new Paragraph({ spacing: { after: 60 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ derived numbers
const S = V.splits;
const nAll = V.total_rows;
const inv = A.inventory, L = A.labels, AN = A.annotators, TX = A.text, ID = A.identity, M = A.metadata;
const tac = AN.toxicity_annotator_count, band = AN.toxic_rate_by_annotator_band, tacRel = tac.relation;
const corrL = A.label_corr_pearson;
const stt = TX.stats, rel = (c) => stt[c].relation;
const fin = C.final, fun = C.funnel, dec = C.decisions;
const kept = dec.text_stats.kept, dropped = dec.text_stats.dropped;
const pubCats = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "clean", "preprocessing_params.json"))).publication_one_hot.categories;
const nPubCols = pubCats.length + 1;
const nTab = fin.n_features_tabular;
const uniqRange = (cols) => { const v = cols.map((c) => inv[c].n_unique); return `${N(Math.min(...v))}–${N(Math.max(...v))}`; };
const LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"];
const IDS = Object.keys(ID.per_identity);
const REACT = ["funny", "wow", "sad", "likes", "disagree"];
const maxReactRho = Math.max(...REACT.map((c) => Math.abs(M.reactions[c].relation.spearman_score)));
const pubRates = M.publication_id.toxic_rate_range_n_ge_1000;
const cv = M.created_date.info_gain;
const HY = M.label_entropy_bits;
const IGb = (v) => `${F(v, 4)} bits, ${F(100 * v / HY, 2)}% of H(y)`;
const CA = K.cross_corr_all;
const caAbove = (() => { const r = []; for (let i = 0; i < CA.names.length; i++) for (let j = i + 1; j < CA.names.length; j++) if (Math.abs(CA.matrix[i][j]) > 0.5) r.push([CA.names[i], CA.names[j]]); return r; })();
const caAllStats = caAbove.every(([a, b]) => a.startsWith("f_n_") || a === "f_mean_word_len" || a === "f_upper_ratio" ? (b.startsWith("f_n_") || b === "f_mean_word_len" || b === "f_upper_ratio") : false);
const subNo = Object.values(A.subtype_also_toxic_pct);
const sk = K.skew;
const mxr = K.max_abs_r;
const worst = K.worst_abs_r_any_feature;
const insultR = corrL.toxicity.insult;
const perId = ID.per_identity;
const nb = W2.neighbours;
const corrPair = K.cross_corr_top_pairs[0];

// ================================================================== CONTENT
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 110 },
    children: [new TextRun({ text: "Detecting Toxic Comments in Online Discussions:", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
    children: [new TextRun({ text: "Preparing the Civil Comments Data", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Moaz Allam, Kareem Elhenawy", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Department of Computer Science and Engineering, The American University in Cairo", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Moaz_Allam@aucegypt.edu, karimelhenawy@aucegypt.edu", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "CSCE 3602 / DSCI 3415 Class Project, Phase 2: Data Preparation, Cleaning and Feature Generation", size: 19, italics: true })] }),
];

const s2 = [];
s2.push(new Paragraph({ spacing: { after: 50 }, children: [new TextRun({ text: "ABSTRACT", bold: true })] }));
firstAfterHeading = true;
s2.push(body(
  `This report prepares the Civil Comments corpus for the models of the next phase. Phase 1 used an 8-column copy of the data; here we use the full 45-column release. We analysed every column on the training split for type, missing values, distinct values, distribution and relation to the label. The comment text is the only input that both carries signal and exists when a comment is posted. Two columns that look predictive are not legitimate inputs: the number of annotators, whose Spearman correlation with the toxicity score is ${F(tacRel.spearman_score)} (${P(band["3-5"].toxic_rate)} of comments seen by 3–5 annotators are toxic, against ${P(band["10-99"].toxic_rate, 1)} of those seen by 10–99), and the platform's own moderation verdict. The identity columns are missing for ${Pp(inv.male.pct_missing)} of comments, and not at random. Cleaning normalised the text and removed ${N(dec.duplicates.train_rows_removed)} duplicate training rows and ${N(dec.cross_split_leakage.train_rows_removed)} training rows whose text also appears in validation or test, leaving ${N(fin.rows.train)}, ${N(fin.rows.validation)} and ${N(fin.rows.test)} comments. Each comment is represented by a ${W2.settings.vector_size}-dimension Word2Vec embedding of its text and ${nTab} scaled tabular features. No feature correlates with any target beyond |r| = ${F(worst, 3)}.`));
s2.push(new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: "KEYWORDS  ", bold: true }), new TextRun({ text: "data cleaning, feature engineering, missing values, data leakage, word2vec, toxic comment classification" })] }));

// ---------------------------------------------------------------- 1 INTRODUCTION
s2.push(h1(1, "Introduction"));
s2.push(body(`Our project builds a moderation aid: a model that reads a comment and scores it on six toxicity attributes, so that a human moderator can read the worst comments first. Phase 1 surveyed the literature and the available datasets and chose Civil Comments [1]. It also fixed two decisions that this phase builds on: a comment counts as positive for an attribute when at least half of its annotators said so (score ≥ 0.5), and the task is multi-label, since one comment can be both an insult and an identity attack.`));
s2.push(body(`This phase turns the raw release into data a model can use. We describe the dataset and why we chose it (Section 2), analyse every column (Section 3), clean and transform the data (Section 4), and list the final features with the size and balance of the cleaned dataset (Section 5). Three rules run through the whole phase. Every statistic and figure comes from a script, and the scripts are submitted with this report. Every relation to the label and every fitted parameter (scaler means, encoder categories, the embedding model) is computed on the training split only, so nothing from validation or test leaks into a decision. And a column becomes a feature only if it would be available at the moment a new comment is posted, because that is when the moderation aid has to score it.`));
s2.push(body(`The scripts run in order: \`00_download\` fetches the data and checks file hashes, \`01_load\` loads the three files into one table, \`02_feature_analysis\` produces the analysis in Section 3, \`03_clean_preprocess\` cleans, encodes and scales the data, \`04_word2vec\` embeds the text, \`05_correlation_checks\` measures the final features, and \`06_make_figures\` draws every figure.`));

// ---------------------------------------------------------------- 2 DATASET
s2.push(h1(2, "The Dataset"));
s2.push(h2("2.1", "Description"));
s2.push(body(`Civil Comments was a commenting service used by independent news sites. When it closed at the end of 2017 its archive of public comments was released, and Jigsaw had the comments rated for toxicity; the result was published for the Kaggle competition "Jigsaw Unintended Bias in Toxicity Classification" [3] and is in the public domain (CC0) [4]. The comments were posted between ${day(M.created_date.min)} and ${day(M.created_date.max)} on ${inv.publication_id.n_unique} publications.`));
s2.push(body(`Each comment was shown to a panel of crowd raters, and each of the seven attributes (toxicity, severe_toxicity, obscene, threat, insult, identity_attack, sexual_explicit) is stored as the fraction of raters who said it applied. A subset of comments was also shown to raters who marked which of 24 identities the comment mentions, in five groups: gender, sexual orientation, religion, race or ethnicity, and disability [2, 3].`));
s2.push(body(`The release comes as three files, which we use as three splits: the training file (${N(S.train.rows)} comments), the public test file as validation (${N(S.validation.rows)}) and the private test file as test (${N(S.test.rows)}). We found that the files are split by article: none of the ${N(M.article_id.n_unique)} articles in the training split has a comment in validation or test. Comments in one thread share a topic, vocabulary and quoted text, so a split that ignored articles would show a model half of a conversation in training and the other half at evaluation. We therefore keep the official split.`));
s2.push(h2("2.2", "Source"));
s2.push(body(`Phase 1 used \`google/civil_comments\` [4], which has only the text and the seven scores. Phase 2 needs every column, which only the original competition files have. Kaggle requires a signed-in account to download them, so we took the three competition files from a public Hugging Face mirror [7].`));
s2.push(h2("2.3", "Why Civil Comments"));
s2.push(body(`Phase 1 compared five public datasets and chose Civil Comments for three reasons [1]. It is the largest, at about two million comments. Its labels are fractions, so a comment eight of ten raters called toxic is distinguishable from one that five of ten did. And it is the only dataset of this size with identity annotations, which our planned bias audit needs [2]. The full release adds a fourth reason that matters for this phase: besides free text it has numeric, nominal, temporal and identifier columns, and real missing values, so every preparation step has something to act on.`));

// ---------------------------------------------------------------- 3 FEATURE ANALYSIS
s2.push(h1(3, "Feature Analysis"));
s2.push(body(`Table 1 lists all ${V.n_columns} columns with their type, missing values, number of distinct values and what this phase does with them. All relations to the label in this section are measured on the ${N(A.n_train)} training comments, and the label is either the toxicity score or its binary form y = (toxicity ≥ 0.5). For a numeric column we report the Pearson and Spearman correlations with the score, Pearson's r with the 0/1 label y, and the column's mean in each class. For a nominal column we report the toxic rate in each category and the information gain IG = H(y) − H(y | column), the entropy-based measure that ID3 uses to choose decision-tree splits. The label's entropy is H(y) = ${F(HY, 3)} bits, so IG says how much of that uncertainty a column removes.`));

const s3 = [];
s3.push(...table(`All ${V.n_columns} columns of the release (missing values and distinct values counted over all ${N(nAll)} rows).`,
  [1900, 560, 1300, 1550, 1250, 3520],
  ["Column(s)", "#", "Type", "Missing", "Distinct values", "What Phase 2 does with it"],
  [
    ["id", "1", "integer key", "0", N(inv.id.n_unique), "kept as the row key"],
    ["comment_text", "1", "free text", "0 (0 empty)", N(inv.comment_text.n_unique), "cleaned; encoded as a Word2Vec embedding and 12 statistics"],
    ["toxicity and 6 subtypes", "7", "fraction in [0, 1]", "0", uniqRange(LABELS), "targets, binarised at 0.5; severe_toxicity is not a target"],
    ["identity columns", "24", "fraction in [0, 1]", `${N(inv.male.n_missing)} (${Pp(inv.male.pct_missing)})`, uniqRange(IDS), "bias audit only; not imputed; flag added"],
    ["identity_annotator_count", "1", "count", "0", N(inv.identity_annotator_count.n_unique), "dropped: duplicates the flag"],
    ["toxicity_annotator_count", "1", "count", "0", N(inv.toxicity_annotator_count.n_unique), "not a feature (leaks the label); kept for reference"],
    ["created_date", "1", "timestamp (UTC)", "0", N(inv.created_date.n_unique), "not a feature: no relation to the label"],
    ["publication_id", "1", "nominal", "0", N(inv.publication_id.n_unique), `one-hot, ${pubCats.length} columns + "other"`],
    ["parent_id", "1", "identifier", `${N(inv.parent_id.n_missing)} (${Pp(inv.parent_id.pct_missing)})`, N(inv.parent_id.n_unique), "replaced by is_reply"],
    ["article_id", "1", "identifier", "0", N(inv.article_id.n_unique), "dropped: an identifier no new comment shares"],
    ["rating", "1", "nominal (2 values)", "0", N(inv.rating.n_unique), "not a feature: a verdict given after posting"],
    ["funny, wow, sad, likes, disagree", "5", "count", "0", uniqRange(REACT), "dropped: given after posting, no signal"],
  ]));

const s4 = [];
s4.push(h2("3.1", "Labels"));
s4.push(body(`${Fg("labeldist")} shows the distribution of each score over ten bins. The scores are dominated by zeros: ${Pp(L.toxicity.pct_zero)} of training comments have a toxicity score of exactly 0, and between ${Pp(Math.min(...LABELS.slice(1).map((c) => L[c].pct_zero)))} and ${Pp(Math.max(...LABELS.slice(1).map((c) => L[c].pct_zero)))} have 0 on each subtype. Because panels differ in size, the scores take many distinct values (${N(L.toxicity.n_distinct_values)} for toxicity), not just multiples of one tenth. At the 0.5 threshold, toxicity applies to ${N(L.toxicity["positives_at_0.5"])} comments (${Pp(L.toxicity["pct_positive_at_0.5"])}), insult to ${N(L.insult["positives_at_0.5"])} (${Pp(L.insult["pct_positive_at_0.5"])}), identity_attack to ${N(L.identity_attack["positives_at_0.5"])}, obscene to ${N(L.obscene["positives_at_0.5"])}, sexual_explicit to ${N(L.sexual_explicit["positives_at_0.5"])}, threat to ${N(L.threat["positives_at_0.5"])}, and severe_toxicity to ${N(L.severe_toxicity["positives_at_0.5"])}. A classifier that always answers "no" would be ${Pp(fin.majority_baseline_acc_train.toxicity)} accurate on toxicity and over ${Pp(Math.min(...["obscene", "threat", "identity_attack", "sexual_explicit"].map((c) => fin.majority_baseline_acc_train[c])), 1)} accurate on the rare subtypes, which is why accuracy will not be a headline metric.`));
s4.push(...figure("labeldist", "fig0_label_distribution.png", "Distribution of the seven label scores over the training split: percentage of comments in each tenth of the score range (colour on a log scale). The orange line is the 0.5 threshold; everything to its right counts as positive."));
s4.push(body(`${Fg("labels")} shows how the seven scores relate. Insult and toxicity move together almost perfectly (r = ${F(insultR)}), and the other subtypes correlate with toxicity at r = ${F(Math.min(...["severe_toxicity", "obscene", "threat", "identity_attack", "sexual_explicit"].map((c) => corrL.toxicity[c])))} to ${F(Math.max(...["severe_toxicity", "obscene", "threat", "identity_attack", "sexual_explicit"].map((c) => corrL.toxicity[c])))}. At the 0.5 threshold, between ${Pp(Math.min(...subNo), 1)} and ${Pp(Math.max(...subNo), 1)} of the comments that carry a subtype are also toxic, yet ${Pp(A.toxic_without_any_subtype_pct)} of toxic comments carry no subtype at all. Most comments with any label carry exactly two (${N(A["labels_per_comment_at_0.5"]["2"])} comments) rather than one (${N(A["labels_per_comment_at_0.5"]["1"])}). This is the multi-label structure that calls for one sigmoid output per attribute. The r = ${F(insultR)} between insult and toxicity is above the 0.85 limit for leakage, which is a second reason the scores are only ever outputs: no score is used as an input to predict another.`));
s4.push(...figure("labels", "fig1_label_correlation.png", `Pearson correlation between the seven label scores (training split). Insult is almost a copy of toxicity; severe_toxicity has ${N(L.severe_toxicity["positives_at_0.5"])} positives at 0.5 and is not a modelling target.`));

s4.push(h2("3.2", "Annotator counts: a leak, not a feature"));
s4.push(body(`toxicity_annotator_count ranges from ${N(tac.min)} to ${N(tac.max)} with a median of ${N(tac.median)}; ${N(tac.top_values["4"])} training comments (${P(tac.top_values["4"] / A.n_train, 1)}) had four annotators and ${N(tac.top_values["10"])} had ten. Its Spearman correlation with the toxicity score is ρ = ${F(tacRel.spearman_score)}, above the 0.85 limit at which a column stands in for the label (Pearson's r is only ${F(tacRel.pearson_score)}, because a few comments had thousands of annotators). It is the only column in the release that crosses the limit. ${Fg("annot")} shows why. Comments that were seen by 3–5 annotators are toxic ${P(band["3-5"].toxic_rate)} of the time; comments seen by 10–99 annotators are toxic ${P(band["10-99"].toxic_rate, 1)} of the time.`));
s4.push(body(`The dataset documentation notes that some comments were shown to far more raters than the usual panel [3]. Whatever the rule, the count is decided by the annotation process after the comment exists, and it depends on how toxic the comment looked. A new comment arriving at the moderation aid has no annotator count. A model trained with it would learn that "many annotators means toxic" and would have nothing to apply that rule to in use, which is leakage. It is therefore not a feature. The cleaned data keeps it as \`n_annotators\` for reference only. identity_annotator_count is 0 for ${Pp(AN.identity_annotator_count.pct_zero)} of training comments and is non-zero exactly when the identity columns are present (checked on every row), so it adds nothing beyond a flag and is dropped.`));
s4.push(...figure("annot", "fig2_annotators_vs_label.png", "Share of training comments that are toxic, by how many annotators rated them. The count is set by the annotation process, not by the comment, and is unavailable for a new comment."));

s4.push(h2("3.3", "Comment text"));
s4.push(body(`No comment is missing or empty. Across all splits there are ${N(inv.comment_text.n_unique)} distinct texts in ${N(nAll)} rows. Training comments have a median of ${N(stt.n_chars.median)} characters and ${N(stt.n_words.median)} words, a mean of ${N(stt.n_chars.mean)} characters, and a maximum of ${N(stt.n_chars.max)}; the length distribution is right-skewed (skew ${F(stt.n_chars.skew)}) with a long tail of essays. ${Fg("length")} compares the word-count distributions of toxic and non-toxic comments. They overlap almost completely; toxic comments are slightly shorter on average (${N(rel("n_chars").mean_if_toxic)} against ${N(rel("n_chars").mean_if_not)} characters).`));
s4.push(...figure("length", "fig3_length_by_class.png", "Distribution of words per comment for toxic and non-toxic training comments (each curve sums to 100%). Length alone barely separates the classes."));
s4.push(body(`Reading the raw text turned up several things to clean. Some comments use Windows line breaks, contain invisible control characters, or use typographic variants such as "…" and styled letters that Unicode normalisation turns into plain text; ${N(TX.html_entity_rows)} contain HTML entities such as "&amp;". ${N(TX.html_tag_rows)} training comments contain something shaped like an HTML tag, but on inspection these are people's own markup such as "<sarcasm>" or "<i>", so they are kept as text. ${N(TX.dup_distinct_texts_repeated)} distinct texts occur more than once in the training split, covering ${N(TX.dup_rows_involved)} rows of which ${N(TX.dup_rows_redundant)} are redundant copies; the most repeated are short replies such as "${TX.dup_top[0].text}" (${TX.dup_top[0].count} times) and "${TX.dup_top[1].text}" (${TX.dup_top[1].count}). For ${N(TX.dup_texts_with_conflicting_y)} of these texts the copies disagree on the label. Finally, ${N(TX.val_test_rows_also_in_train.validation)} validation and ${N(TX.val_test_rows_also_in_train.test)} test comments have an exact copy in the training split.`));
s4.push(body(`To see how much can be read from the style of a comment without its words, we counted ${Object.keys(stt).length} statistics per comment: length in characters, letters and words, mean word length, upper-case letters and their ratio to all letters, fully capitalised words of three or more letters ("shouting"), exclamation and question marks, runs of repeated "!" or "?", words masked with asterisks, links, digits, line breaks and non-ASCII characters. ${Fg("stats")} gives their Pearson correlation with the 0/1 label y. Every one is weak. The strongest are masked words (${sgn(rel("n_masked").pearson_y)}), digits (${sgn(rel("n_digits").pearson_y)}) and exclamation marks (${sgn(rel("n_exclaim").pearson_y)}); digits and links lean non-toxic, because comments that cite figures and sources tend to be arguments rather than attacks. The conclusion is that toxicity lives in the words, not the typography, which is why the text is encoded word by word with Word2Vec (Section 4.7). Its dimensions correlate with the labels several times more strongly than any style statistic (Section 5.2).`));
s4.push(...figure("stats", "fig4_text_stats_vs_label.png", `Pearson correlation of each candidate text statistic with the toxic label (training split). Blue: kept; grey: dropped (Section 4.6). The largest |r| is ${F(Math.max(...Object.values(stt).map((x) => Math.abs(x.relation.pearson_y))), 3)}.`));

s4.push(h2("3.4", "Identity columns"));
s4.push(body(`The 24 identity columns are missing together: a comment has all 24 or none (${N(ID.partially_missing_rows)} partially missing rows), and they are present for ${N(ID.n_annotated)} training comments (${Pp(ID.pct_annotated)}). The missingness is not random: the annotated subset is more toxic than the corpus as a whole (${P(ID.toxic_rate_in_annotated_subset)} against ${P(A.toxic_rate_train)}). Inside it, a comment that mentions any identity is toxic ${P(ID.toxic_rate_any_identity, 1)} of the time and one that mentions none ${P(ID.toxic_rate_no_identity, 1)}; the largest correlation of a single identity with the score is r = ${F(Math.max(...Object.values(perId).map((v) => v.pearson_with_score)), 3)}. The identity columns are annotations of the text made after the fact, not something a new comment comes with, so they are never model inputs; they are kept for the bias audit planned in Phase 1 [1, 2].`));

s4.push(h2("3.5", "Metadata"));
s4.push(body(`**rating** records the platform's moderation outcome: ${N(M.rating.values.approved)} training comments were approved and ${N(M.rating.values.rejected)} rejected. Rejected comments are toxic ${P(M.rating.toxic_rate.rejected, 1)} of the time against ${P(M.rating.toxic_rate.approved, 1)} for approved ones (information gain ${IGb(M.rating.info_gain)}, the largest of any metadata column). The verdict was given by people judging the same comment after it was written, so it is a second label rather than an input, and it is not a feature.`));
s4.push(body(`**publication_id** has ${M.publication_id.n_unique} values. The ten largest publications hold ${Pp(M.publication_id.top10_share_pct, 1)} of training comments, while ${M.publication_id.n_with_under_1000_rows} publications have fewer than 1,000 comments each (${N(M.publication_id.rows_in_publications_under_1000)} comments in total). Every publication in validation and test also appears in training. ${Fg("pub")} shows that the toxic rate differs from site to site, from ${P(pubRates[0])} to ${P(pubRates[1])} among publications with at least 1,000 comments (information gain ${IGb(M.publication_id.info_gain)}). The publication is known when a comment is posted, so it is kept as a feature.`));
s4.push(...figure("pub", "fig6_publication_toxic_rate.png", "Share of toxic comments for each publication with at least 1,000 training comments. The orange line is the rate over all training comments."));
s4.push(body(`**created_date** spans ${day(M.created_date.min)} to ${day(M.created_date.max)}, and all three splits cover the same range, so the split is not by time. Only ${N(M.created_date.by_year["2015"].n)} training comments are from 2015. The date carries almost nothing about the label. Split into year, month, hour and day of the week, no part removes more than ${F(100 * Math.max(...Object.values(cv)) / HY, 2)}% of the label's entropy (information gain ${F(cv.year, 6)}, ${F(cv.month, 4)}, ${F(cv.hour_utc, 4)} and ${F(cv.weekday, 4)} bits), and the toxic rate is ${P(M.created_date.by_year["2016"].toxic_rate)} in 2016 and ${P(M.created_date.by_year["2017"].toxic_rate)} in 2017. The year would also never generalise: a model in use after 2017 would only see years it was not trained on.`));
s4.push(body(`**parent_id** is missing for ${Pp(M.parent_id.pct_missing)} of training comments, but these values are not lost: a top-level comment has no parent. The useful information is whether the comment is a reply. Top-level comments are toxic ${P(M.parent_id.toxic_rate_top_level)} of the time and replies ${P(M.parent_id.toxic_rate_reply)} (information gain ${IGb(M.parent_id.info_gain_is_reply)}). The parent comment itself is in the dataset for ${Pp(M.parent_id.pct_parents_present_in_data, 1)} of replies, which a later phase could use as context.`));
s4.push(body(`**article_id** identifies ${N(M.article_id.n_unique)} training articles with a median of ${N(M.article_id.comments_per_article.median)} comments each (maximum ${N(M.article_id.comments_per_article.max)}). Longer threads are slightly more toxic (Spearman ρ = ${F(M.article_id.spearman_thread_size_vs_score, 3)} between thread size and score), but the size of a thread is only known after it has grown, and no article is shared with validation or test. **Reader reactions** (funny, wow, sad, likes, disagree) are zero for ${Pp(Math.min(...REACT.map((c) => M.reactions[c].pct_zero)), 1)} to ${Pp(Math.max(...REACT.map((c) => M.reactions[c].pct_zero)), 1)} of comments, are heavily right-skewed (skew ${F(Math.min(...REACT.map((c) => M.reactions[c].skew)), 1)} to ${F(Math.max(...REACT.map((c) => M.reactions[c].skew)), 1)}), arrive after posting, and barely relate to the label (|ρ| ≤ ${F(maxReactRho, 3)}).`));

s4.push(h2("3.6", "Summary"));
s4.push(body(`Table 2 puts the candidate inputs side by side. The two strongest columns, the annotator count and the moderation verdict, are both produced after a comment is written and would leak the label; the annotator count is also above the 0.85 correlation limit. Of the columns available at posting time, the publication carries a little signal and the style statistics carry little. The reply flag and the date are both very weak (under 0.1% of H(y) each). We keep the reply flag because it is a structural fact about the comment that costs one binary column, and drop the date because none of its parts is stronger and the year cannot generalise. Almost all of the usable information is in the words, which Section 4.7 encodes.`));
s4.push(...table(`Relation of each candidate input to the toxic label (training split) and whether it exists when a comment is posted. IG is information gain in bits, out of H(y) = ${F(HY, 3)}.`,
  [1650, 1700, 650, 800],
  ["Candidate", "Relation to y", "At posting?", "Feature?"],
  [
    ["toxicity_annotator_count", `ρ ${F(tacRel.spearman_score, 3)} (≥ 0.85)`, "no", "no"],
    ["rating", `IG ${F(M.rating.info_gain, 4)}`, "no", "no"],
    ["identity columns", `r ≤ ${F(Math.max(...Object.values(perId).map((v) => v.pearson_with_score)), 3)}`, "no", "no"],
    ["publication_id", `IG ${F(M.publication_id.info_gain, 4)}`, "yes", "yes"],
    ["thread size (article_id)", `ρ ${F(M.article_id.spearman_thread_size_vs_score, 3)}`, "no", "no"],
    ["reactions", `|ρ| ≤ ${F(maxReactRho, 3)}`, "no", "no"],
    ["is_reply (parent_id)", `IG ${F(M.parent_id.info_gain_is_reply, 4)}`, "yes", "yes"],
    ["created_date", `IG ≤ ${F(Math.max(...Object.values(cv)), 4)}`, "yes", "no"],
    ["text statistics", `|r| ≤ ${F(Math.max(...Object.values(stt).map((s) => Math.abs(s.relation.pearson_y))), 3)}`, "yes", "12 of 15"],
    ["words (Word2Vec)", `|r| ≤ ${F(Math.abs(mxr.word2vec.r), 3)}`, "yes", "yes"],
  ], ["l", "l", "l", "l"]));

// ---------------------------------------------------------------- 4 CLEANING
s4.push(h1(4, "Cleaning and Pre-processing"));
s4.push(body(`Cleaning happens in a fixed order. Row-level text cleaning is applied identically to every split, since it needs no fitted parameter. Row filtering is applied to the training split only, so validation and test remain the benchmark rows and results stay comparable with published work on the same split. Every transformation that learns something from the data (feature selection, percentiles, means and standard deviations, one-hot categories and the Word2Vec model) is fitted on the cleaned training split and then applied unchanged to validation and test. Table 3 tracks the number of rows through each step.`));
s4.push(...table("Rows after each cleaning step.",
  [2150, 950, 850, 850],
  ["Step", "Train", "Validation", "Test"],
  fun.map((f, i) => [["Raw release", "A. Text cleaned, empty removed", "B1. Duplicates collapsed", "B2. Overlap with val/test removed"][i] || f.step,
    N(f.train), N(f.validation), N(f.test)]),
  ["l", "r", "r", "r"]));

s4.push(h2("4.1", "Text cleaning"));
s4.push(body(`Each comment is cleaned into two versions. The first, \`text\`, keeps case and punctuation for models that read raw text, such as the transformer planned for Phase 3. It is produced by converting Windows line endings, applying Unicode NFKC normalisation, deleting control and zero-width characters, collapsing runs of spaces and tabs, allowing at most one blank line, trimming, and unescaping HTML entities. These steps changed ${N(C.text_cleaning.rows_changed)} comments (${Pp(C.text_cleaning.pct_rows_changed, 1)}), mostly by collapsing the double space after a full stop, and removed ${N(C.text_cleaning.chars_removed)} characters in total. No comment became empty.`));
s4.push(body(`The second, \`text_bow\`, is the input to Word2Vec. It lower-cases the text; replaces links with the token "xxurl", words masked with asterisks with "xxmasked" and numbers with "xxnum"; expands contractions; and replaces every remaining character that is not a letter with a space. The placeholders keep the fact that a link, a masked swear word or a number was present without creating a separate word for every variant. Contractions are expanded before apostrophes are removed because removing them first turns "he'll" into "hell" and "we'll" into "well", inventing profanity. We do not remove stop words: "you", "your" and "they" are the words that aim an insult at someone. We do not stem either: Word2Vec already places spelling variants of a word next to each other, as the neighbour check in Section 4.7 shows. ${N(C.text_cleaning.rows_with_empty_bow_text)} comments consist only of punctuation or emoji and have an empty \`text_bow\`; they keep their row and receive the zero embedding.`));
s4.push(h2("4.2", "Duplicates and overlap between splits"));
s4.push(body(`After text cleaning, ${N(dec.duplicates.train_rows_in_duplicate_groups)} training rows belong to ${N(dec.duplicates.duplicate_groups)} groups of identical texts. Each group is collapsed to the copy with the most annotators, whose label is the most reliable. This removes ${N(dec.duplicates.train_rows_removed)} rows, including the copies in the ${N(dec.duplicates.groups_with_conflicting_y)} groups whose copies disagree on the label. Then every training row whose text also occurs in validation or test is removed (${N(dec.cross_split_leakage.train_rows_removed)} rows), since a model could otherwise memorise an evaluation answer. Validation and test are not deduplicated, so they remain the official benchmark.`));
s4.push(h2("4.3", "What was not removed"));
s4.push(body(`Only ${N(AN.toxicity_annotator_count.n_below_4)} training comments had fewer than four annotators, and their scores are still exact fractions of their votes, so they stay. Very short comments ("No.", "BS") are real comments and some are toxic, so they stay too. There are no runaway lengths to cap: the longest comment has ${N(stt.n_chars.max)} characters and the 99th percentile is ${N(stt.n_chars.p99)}. We did not resample the data to balance the classes. Copies of rare comments would leak across cross-validation folds, and the test split has to keep the real balance.`));
s4.push(h2("4.4", "Missing values"));
s4.push(body(`Only two columns have missing values (Table 1). For parent_id, the missing value means "top-level comment", so the column is replaced by a binary \`is_reply\` with no missing values. The identity columns are deliberately not imputed. The usual fills (zero, the mean, the most common value) would all claim something about ${Pp(inv.male.pct_missing)} of comments that nobody checked: filling with 0 would assert that no identity is mentioned. Because the missingness is not at random (Section 3.4), even the mean of the annotated subset would be biased. The columns are kept as they are, with a flag \`identity_annotated\`, and the bias audit will use only annotated comments. No other column has a missing value in any split.`));
s4.push(h2("4.5", "Labels"));
s4.push(body(`The seven scores are kept, and a binary target \`y_<attribute>\` = (score ≥ 0.5) is added for each. Six are modelling targets. severe_toxicity is not: with ${N(L.severe_toxicity["positives_at_0.5"])} positives in ${N(A.n_train)} training comments there is nothing to learn, and its score is kept only for completeness. Keeping the scores makes the threshold sensitivity check from Phase 1 possible later at 0.4 and 0.6.`));
s4.push(h2("4.6", "Encoding, transformation and scaling"));
s4.push(body(`**Text statistics.** Of the ${Object.keys(stt).length} candidates, a statistic is dropped if it adds nothing new (|r| > 0.9 with a statistic already kept) or if it has no relation to the label (|ρ| < 0.01 and |r| < 0.01 with y). ${Object.entries(dropped).map(([k, v]) => `${k.replace(/_/g, " ")} is ${v.replace("no association with the label", "unrelated to the label").replace("r_y", "r")}`).join("; ")}. That leaves ${kept.length}. ${Fg("dist")} shows why they need transforming: the counts are piled up at zero with long right tails (skew ${F(sk.n_exclaim.before, 1)} for exclamation marks and ${F(sk.n_words.before, 2)} for words). Each count is replaced by log(1 + x), which pulls the tail in (skew ${F(sk.n_exclaim.after, 2)} and ${F(sk.n_words.after, 2)} afterwards). The two ratios, mean word length and the upper-case ratio, are capped at their 1st and 99th training percentiles instead, since comments made only of a link have mean word lengths above 40. Every statistic is then standardised to z = (x − mean) / std with the training mean and standard deviation.`));
s4.push(body(`Scaling matters because the models planned for Phase 3 are sensitive to it. In k-nearest neighbours, word counts in the hundreds would swamp a ratio between 0 and 1 in every distance. In logistic regression, the SVM and the neural network, features on different scales slow gradient descent and make regularisation punish features unevenly. We use standardisation rather than min-max scaling because one extreme comment would set the maximum and squash every other value into a narrow band.`));
s4.push(...figure("dist", "fig8_distributions_before_after.png", "Three text statistics before (left, raw) and after (right, scaled) their transform and standardisation, training split. Counts get log(1 + x); ratios are capped at the 1st/99th percentiles. Each panel's skew is printed above it."));
s4.push(body(`**Publication.** publication_id is nominal: publication 54 is not "more" than publication 6. Label encoding would invent that order for linear models and KNN, and frequency encoding would merge publications of similar size. We therefore one-hot encode it. The ${pubCats.length} publications with at least 1,000 training comments get a column each; the ${M.publication_id.n_with_under_1000_rows} smaller ones, and any publication never seen in training, share one "other" column, so no column is almost empty and an unknown site still has a valid encoding. That gives ${nPubCols} binary columns. **Reply.** \`is_reply\` is already 0/1 and is not rescaled.`));
s4.push(h2("4.7", "Text representation"));
s4.push(body(`The comment text has to be either one-hot encoded or embedded. A one-hot encoding would need a column for every word in a vocabulary of tens of thousands, almost all zero for any one comment, and would treat "idiot" and "moron" as unrelated columns. We therefore embed the text with Word2Vec, which places words used in similar contexts close together. We trained a skip-gram model [5, 6] on the training comments only, with ${W2.settings.vector_size} dimensions, a window of ${W2.settings.window} words, words seen at least ${W2.settings.min_count} times, ${W2.settings.negative} negative samples and ${W2.settings.epochs} epochs. We chose skip-gram over CBOW because skip-gram learns semantic relationships between words, while CBOW mostly learns syntactic, positional similarity, and toxicity is a matter of meaning: "idiot" and "moron" should be close because they mean the same thing. Skip-gram is slower to train and needs more data, which a training corpus of this size provides. The model learned ${N(W2.vocabulary_size)} words from ${N(W2.corpus_words)} training tokens, and ${Pp(W2.coverage.validation.pct_tokens_in_vocabulary, 1)} of validation tokens are in its vocabulary. A quick check confirms that it learned meaning: the nearest neighbours of "idiot" are ${nb.idiot.slice(0, 4).map((x) => `"${x[0]}"`).join(", ")}, and those of "thanks" are ${nb.thanks.slice(0, 4).map((x) => `"${x[0]}"`).join(", ")}. A comment's embedding is the mean of its words' vectors (the zero vector for the ${N(W2.coverage.train.comments_with_no_known_word)} training comments with no known word). Each of the ${W2.settings.vector_size} dimensions is then standardised with the training mean and standard deviation, like the other numeric features.`));

// ---------------------------------------------------------------- 5 FINAL
const s5 = [];
s5.push(h1(5, "Final Features and Dataset"));
s5.push(body(`Table 4 lists the final feature set and Table 5 the size and balance of the cleaned data. Before accepting the features we ran two checks on the training split.`));

const s6 = [];
s6.push(...table("The final feature set. All parameters are fitted on the cleaned training split.",
  [1750, 900, 3530, 3900],
  ["Block", "Columns", "Content", "Encoding and scaling"],
  [
    ["Words (Word2Vec)", String(W2.settings.vector_size), "mean skip-gram vector of the comment's words", "z-score per dimension"],
    ["Text statistics", String(kept.length), kept.map((k) => k.replace(/_/g, " ")).join(", "), "counts: log(1 + x); ratios: capped at p1/p99; then z-score"],
    ["Reply flag", "1", "is_reply (from parent_id)", "binary 0/1"],
    ["Publication", String(nPubCols), `${pubCats.length} publications + "other"`, "one-hot; rare or unseen publications go to \"other\""],
    ["Targets", "6", "y_toxicity, y_insult, y_identity_attack, y_obscene, y_threat, y_sexual_explicit", "score ≥ 0.5; the seven scores are kept"],
    ["Kept, not inputs", "—", "n_annotators, 24 identity scores + identity_annotated (bias audit), text, text_bow, publication_id, created_date, rating", "unchanged; never given to a model as input"],
    ["Removed", "—", "identity_annotator_count, parent_id, article_id, five reaction counts, n_chars, n_letters, n_non_ascii", "reasons in Sections 3 and 4.6"],
  ]));

const s7 = [];
s7.push(h2("5.1", "Feature-to-feature correlation"));
s7.push(body(`${Fg("xcorr")} is the correlation matrix of the ${K.cross_corr_core.features.length} scaled tabular features. The strongest pair is ${corrPair[0].slice(2).replace(/_/g, " ")} with ${corrPair[1].slice(2).replace(/_/g, " ")} (r = ${F(corrPair[2], 2)}), below the 0.9 redundancy cut. Every other pair is weaker, so each statistic adds something its neighbours do not. The publication columns correlate with each other ${K.cross_corr_publication_all_negative ? "only negatively" : "mostly negatively"}, as one-hot columns of a single variable must: a comment belongs to one publication, so a 1 in one column means 0 in the rest. The strongest pair is publications ${K.cross_corr_publication_strongest[0].slice(6)} and ${K.cross_corr_publication_strongest[1].slice(6)} (r = ${F(K.cross_corr_publication_strongest[2], 2)}), the two largest sites. Their relation to the other features is weak (largest |r| = ${F(K.cross_corr_publication_vs_core_max, 2)}).`));
s7.push(...figure("xcorr", "fig9_feature_cross_correlation.png", "Pearson correlation between the scaled tabular features (training split). Blue is positive, orange negative."));
s7.push(body(`${Fg("allcorr")} extends the check to all ${CA.n_features} final features by adding the ${CA.n_word2vec} Word2Vec dimensions. Of the ${N(CA.n_pairs)} pairs of features, none exceeds 0.9 and only ${CA["n_pairs_above_0.5"]} exceed 0.5${caAllStats ? ", all of them pairs of text statistics already shown above" : ""}. The strongest pair of Word2Vec dimensions has r = ${F(CA.w2v_vs_w2v[2], 2)} (${CA.w2v_vs_w2v[0].replace("w2v_", "dimensions ")} and ${CA.w2v_vs_w2v[1].replace("w2v_", "")}), and the strongest relation between a Word2Vec dimension and a tabular feature is r = ${F(CA.w2v_vs_tabular[2], 2)} (${CA.w2v_vs_tabular[0].replace("w2v_", "dimension ")} with ${CA.w2v_vs_tabular[1].slice(2).replace(/_/g, " ")}). No feature is a copy of another, and the embedding carries information that the tabular features do not, so no feature is removed for redundancy.`));
s7.push(...figure("allcorr", "fig11_all_feature_correlation.png", `Pearson correlation between all ${CA.n_features} final features (training split): ${CA.n_tabular - nPubCols - 1} text statistics, the reply flag, ${nPubCols} publication columns and ${CA.n_word2vec} Word2Vec dimensions. Blue is positive, orange negative; the diagonal is each feature with itself.`));
s7.push(h2("5.2", "Correlation with the labels and the leakage check"));
s7.push(body(`A feature that correlates with a target at |r| ≥ 0.85 would be standing in for the label rather than helping to predict it, so any such feature has to be dropped. \`05_correlation_checks\` computes Pearson's r between every final feature and each of the six targets on the training split, and stops with an error if any value reaches 0.85. ${Fg("ycorr")} shows the values for the tabular features. The largest absolute correlations are ${F(Math.abs(mxr.tabular.r), 3)} for the tabular features (${mxr.tabular.feature.slice(2).replace(/_/g, " ")} with ${mxr.tabular.target}), and ${F(Math.abs(mxr.word2vec.r), 3)} for the ${W2.settings.vector_size} Word2Vec dimensions (${mxr.word2vec.feature} with ${mxr.word2vec.target}). All are far below 0.85, so no feature is removed by this rule. The only pair in the data that does exceed it is two labels, insult and toxicity (r = ${F(insultR, 2)}), and labels are never inputs. Before cleaning, the one input column above the limit was toxicity_annotator_count (ρ = ${F(tacRel.spearman_score)}), which Section 3.2 removed.`));
s7.push(...figure("ycorr", "fig10_feature_target_correlation.png", `Pearson correlation between each scaled tabular feature and each of the six targets (training split). The leakage limit is |r| = 0.85; the largest value is ${F(Math.abs(mxr.tabular.r), 3)}.`));
s7.push(h2("5.3", "The cleaned dataset"));
const pq = (v) => (v > 0 && v < 0.01 ? Pp(v, 4) : Pp(v));
const yrow = (c) => [c.replace(/_/g, " "), `${N(fin.positives.train[c].count)} (${pq(fin.positives.train[c].pct)})`,
  `${N(fin.positives.validation[c].count)} (${pq(fin.positives.validation[c].pct)})`, `${N(fin.positives.test[c].count)} (${pq(fin.positives.test[c].pct)})`];
s7.push(...table("The cleaned dataset: comments and positives per target at 0.5.",
  [1300, 1300, 1100, 1100],
  ["", "Train", "Validation", "Test"],
  [["comments", N(fin.rows.train), N(fin.rows.validation), N(fin.rows.test)],
    ...["toxicity", "insult", "identity_attack", "obscene", "sexual_explicit", "threat", "severe_toxicity"].map(yrow),
    ["identity annotated", N(fin.identity_annotated.train), N(fin.identity_annotated.validation), N(fin.identity_annotated.test)]],
  ["l", "r", "r", "r"]));
s7.push(body(`The cleaned data holds ${N(fin.rows.all)} comments: ${N(fin.rows.train)} for training, ${N(fin.rows.validation)} for validation and ${N(fin.rows.test)} for testing. Cleaning did not change the class balance, which stays at ${Pp(fin.positives.train.toxicity.pct)} toxic in training and ${Pp(fin.positives.validation.toxicity.pct)} and ${Pp(fin.positives.test.toxicity.pct)} in validation and test. The mean comment is ${N(fin.text_chars.train.mean)} characters (median ${N(fin.text_chars.train.median)}). Each split is stored as a Parquet file of ${fin.n_columns} columns (${MB(fin.files["train.parquet"])} for training) with its row-aligned Word2Vec matrix. Every fitted parameter (clip limits, means, standard deviations, publication categories and the word vectors) is saved next to them, so the same pipeline can be applied to a new comment.`));
if (DRIVE_URL) s7.push(new Paragraph({ spacing: { before: 60, after: 60 }, children: [
  new TextRun({ text: "Cleaned dataset: ", bold: true }),
  new ExternalHyperlink({ link: DRIVE_URL, children: [new TextRun({ text: DRIVE_URL, color: "1F4E9A", underline: {}, size: SMALL })] }),
] }));

s7.push(h1(6, "Conclusion"));
s7.push(body(`The full Civil Comments release has ${V.n_columns} columns, but the analysis shows that one of them, the comment text, does nearly all the work. The columns that looked most predictive were produced after posting: the annotator count, the moderation verdict, reader reactions and thread size. Using them would have meant training on the answer. The date carries no signal, and the identity columns are missing for most comments and are kept for auditing rather than prediction. Cleaning removed duplicate and leaking training rows without touching the benchmark splits. It left ${N(fin.rows.train)} training comments represented by a Word2Vec embedding and ${nTab} scaled tabular features, none of which approaches the leakage limit. Phase 3 can start training immediately: the features are encoded and scaled, and the bias audit has its identity columns.`));

s7.push(plainHead("AI USAGE DISCLOSURE"));
s7.push(body(`This report was completed with assistance from Claude (Anthropic). The AI was used to write analysis, and feature-generation scripts, to derive the statistics and draw the figures from their outputs. All numbers in the report are read from the outputs of the submitted scripts, the decisions were reviewed by the authors, and the authors are accountable for the accuracy of the content.`));

s7.push(plainHead("REFERENCES"));
const refs = [
  "Moaz Allam and Kareem Elhenawy. 2026. Detecting Toxic Comments in Online Discussions: A Literature Review and Dataset Survey. CSCE 3602 / DSCI 3415 class project, Phase 1 report. The American University in Cairo.",
  "Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. In Companion Proceedings of The 2019 World Wide Web Conference. arXiv:1903.04561",
  "cjadams, Daniel Borkan, inversion, Jeffrey Sorensen, Lucas Dixon, Lucy Vasserman, and nithum. 2019. Jigsaw Unintended Bias in Toxicity Classification. Kaggle. kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification",
  "Google. civil_comments. Hugging Face Datasets, CC0-1.0. huggingface.co/datasets/google/civil_comments",
  "Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781",
  "Radim Řehůřek and Petr Sojka. 2010. Software Framework for Topic Modelling with Large Corpora. In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. ELRA, 45-50.",
  "shuttie. jigsaw-unintended-bias (mirror of the Kaggle competition files). Hugging Face Datasets. huggingface.co/datasets/shuttie/jigsaw-unintended-bias",
];
refs.forEach((r, i) => s7.push(new Paragraph({ spacing: { after: 16, line: 176 }, indent: { left: 230, hanging: 230 },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, size: TINY })] })));

// ================================================================== DOCUMENT
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } };
const two = { count: 2, space: 475, equalWidth: true };
const doc = new Document({
  creator: "Moaz Allam, Kareem Elhenawy",
  title: "Detecting Toxic Comments: Preparing the Civil Comments Data",
  styles: { default: { document: { run: { font: "Times New Roman", size: BODY } } } },
  sections: [
    { properties: { page }, children: title },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: s2 },
    { properties: { page, type: SectionType.CONTINUOUS }, children: s3 },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: [...s4, ...s5] },
    { properties: { page, type: SectionType.CONTINUOUS }, children: s6 },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: s7 },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(DEST, buf);
  console.log("wrote", DEST, buf.length, "bytes;", figNo, "figures,", tabNo, "tables");
});
