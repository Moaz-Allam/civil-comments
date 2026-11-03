/*
 * build_report.js -- writes the Phase 3 report (ACM two-column layout, same page setup as the
 * Phase 1 and Phase 2 reports). Every number is read from report/out/numbers.json, which
 * report/collect.py builds from outputs/*.json and outputs/top3/*.json.
 *
 * Run:  python report/collect.py && python report/make_figures.py && node report/build_report.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, SectionType, BorderStyle, ShadingType,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const DEST = path.join(__dirname, "..", "Phase3_Model_Comparison_Report.docx");
const NUM = JSON.parse(fs.readFileSync(path.join(__dirname, "out", "numbers.json"), "utf8"));
const M = NUM.models;

// ------------------------------------------------------------------ number formatting
const N = (x) => Math.round(Number(x)).toLocaleString("en-US");
const F = (x, d = 3) => Number(x).toFixed(d);
const P = (frac, d = 1) => (100 * Number(frac)).toFixed(d) + "%";
const mins = (s) => (s < 90 ? `${Math.round(s)} s` : `${(s / 60).toFixed(s < 600 ? 1 : 0)} min`);

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
const FIGS = ["all", "cv", "cut", "subgroups"];
FIGS.forEach((k, i) => (figRef[k] = i + 1));
const Fg = (k) => `Figure ${figRef[k]}`;

// ------------------------------------------------------------------ tables
let tabNo = 0;
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
const TARGETS = ["toxicity", "insult", "identity_attack", "obscene", "threat", "sexual_explicit"];
const NAME = { linear_regression: "Linear regression", logistic_regression: "Logistic regression", perceptron: "Perceptron",
  naive_bayes: "Naive Bayes", decision_tree_id3: "ID3", decision_tree_cart: "CART", decision_tree_c45: "C4.5",
  random_forest: "Random forest", svm_linear: "Linear SVM", svm_rbf: "RBF SVM", mlp: "MLP", adaboost: "AdaBoost", knn: "KNN" };
const FEAT = { word2vec: "Word2Vec", doc2vec_dbow: "PV-DBOW", doc2vec: "PV-DM" };
const ORDER = ["linear_regression", "logistic_regression", "perceptron", "naive_bayes", "decision_tree_id3",
  "decision_tree_cart", "decision_tree_c45", "random_forest", "svm_linear", "svm_rbf", "mlp", "adaboost", "knn"].filter((m) => M[m]);
const TOP3 = ["svm_rbf", "mlp", "logistic_regression"];
const tox = (m, s = "validation") => M[m].labels.toxicity[s];
const mean = (m, s = "validation") => M[m].mean[s];
const p0 = (m) => M[m].params.toxicity;
const avgSec = (m) => TARGETS.reduce((a, t) => a + M[m].seconds[t], 0) / TARGETS.length;
const rows = (m) => M[m].n_train.toxicity;
const count = (m, fn) => TARGETS.filter((t) => fn(M[m].params[t], t)).length;
const leader = (metric, t) => ORDER.reduce((b, m) => (M[m].labels[t].validation[metric] > M[b].labels[t].validation[metric] ? m : b), ORDER[0]);
const leads = (m, metric) => TARGETS.filter((t) => leader(metric, t) === m);
const FC = NUM.features;
const PR = NUM.prep, DM = PR.pv_dm, DB = PR.pv_dbow, DS = DB.settings;
const pos = Object.fromEntries(TARGETS.map((t) => [t, PR.positives_pct[t] / 100]));

// one "Performance" paragraph per model, all numbers from the result files
function perf(m, extra = "") {
  const v = tox(m), t = tox(m, "test"), mv = mean(m);
  return `**Performance.** Toxicity (${FEAT[M[m].best_features]}): F1 ${F(v.f1)}, precision ${F(v.precision)}, recall ${F(v.recall)}, ROC-AUC ${F(v.roc_auc)}. Mean over the six labels: F1 ${F(mv.f1)}, ROC-AUC ${F(mv.roc_auc)}. Test toxicity: F1 ${F(t.f1)}, ROC-AUC ${F(t.roc_auc)}.${extra}`;
}

// ================================================================== CONTENT
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 110 },
    children: [new TextRun({ text: "Detecting Toxic Comments in Online Discussions:", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
    children: [new TextRun({ text: "A Pilot Comparison of Supervised Models", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Moaz Allam, Kareem Elhenawy", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Department of Computer Science and Engineering, The American University in Cairo", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Moaz_Allam@aucegypt.edu, karimelhenawy@aucegypt.edu", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "CSCE 3602 / DSCI 3415 Class Project, Phase 3: Pilot Study of Supervised Models", size: 19, italics: true })] }),
];

const A = [];   // two-column text before the wide tables
A.push(new Paragraph({ spacing: { after: 50 }, children: [new TextRun({ text: "ABSTRACT", bold: true })] }));
firstAfterHeading = true;
A.push(body(ABSTRACT()));
A.push(new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: "KEYWORDS  ", bold: true }), new TextRun({ text: "toxic comment classification, model comparison, support vector machine, neural network, class imbalance, result variability" })] }));

// ---------------------------------------------------------------- 1 INTRODUCTION
A.push(h1(1, "Introduction"));
A.push(body(`Our project builds a moderation aid: a model that reads a comment and scores it on six toxicity attributes so that a moderator can read the worst comments first. Phase 1 chose the Civil Comments dataset and framed the task as multi-label classification: a comment is positive for an attribute when at least half of its raters said so [1]. Phase 2 cleaned the data and turned each comment into 133 scaled features [2].`));
A.push(body(`This phase is a pilot study. We train every supervised model covered in the course on the same features and splits, describe how each was set up and tuned, and comment on how well it fits the problem (Section 3). We then take the three best models, compare them in more depth, and measure how much their results move when the data sample, the random seed, the label definition, the decision cut or the group of people a comment mentions changes (Section 4). Section 5 chooses one technique for the rest of the project.`));
A.push(body(`Two changes from the earlier plans apply here. Following the feedback on Phase 2, the text is also encoded with Doc2Vec, which learns one vector for the whole comment instead of averaging word vectors. And the Phase 1 plan to fine-tune a transformer is dropped, because the project is limited to the models taught in the course.`));

// ---------------------------------------------------------------- 2 SETUP
A.push(h1(2, "Experimental Setup"));
A.push(h2("2.1", "Task, data and features"));
A.push(body(`**Labels.** Six binary targets: toxicity, insult, identity_attack, obscene, threat and sexual_explicit, each positive when its score is at least 0.5. One model is trained per label. **Splits.** The cleaned Phase 2 splits: ${N(PR.rows.train)} training, ${N(PR.rows.validation)} validation and ${N(PR.rows.test)} test comments. Positives in training: toxicity ${P(pos.toxicity)}, insult ${P(pos.insult)}, identity_attack ${P(pos.identity_attack, 2)}, obscene ${P(pos.obscene, 2)}, sexual_explicit ${P(pos.sexual_explicit, 2)} and threat ${P(pos.threat, 2)}.`));
A.push(body(`**Features.** Each comment has ${PR.n_tabular} scaled tabular features from Phase 2 (text statistics, a reply flag, publication one-hot) and ${DS.vector_size} text dimensions. We compare three versions of the text dimensions: the mean of the comment's Word2Vec vectors from Phase 2 [6], and the two Doc2Vec models [4], PV-DM and PV-DBOW. PV-DM predicts a word from its neighbours plus a paragraph vector; PV-DBOW predicts the comment's words from the paragraph vector alone, ignoring word order. Both Doc2Vec models were trained on the training split only (${DS.vector_size} dimensions, window ${DS.window}, words seen at least ${DS.min_count} times, ${DS.epochs} epochs) with Gensim [8]. Training vectors stored during Doc2Vec training are not the same as vectors inferred later (mean cosine similarity ${F(DM.cosine, 2)} for PV-DM and ${F(DB.cosine, 2)} for PV-DBOW), so all three splits use inferred vectors, produced in the same way. Both new feature sets pass the Phase 2 leakage rule: the largest |r| between a text dimension and a label is ${F(DM.max_r)} for PV-DM, ${F(DB.max_r)} for PV-DBOW and ${F(PR.word2vec_max_r)} for Word2Vec, far below 0.85.`));
A.push(body(`Table 1 compares the three text versions on the four models that ran on all of them. PV-DBOW and Word2Vec are level (mean ROC-AUC ${F(FC.doc2vec_dbow.mean_auc)} and ${F(FC.word2vec.mean_auc)}), and both beat PV-DM (${F(FC.doc2vec.mean_auc)}). This agrees with Lau and Baldwin, who found that the simpler PV-DBOW is the better Doc2Vec model [5]. PV-DM was therefore dropped, and every other model was trained on Word2Vec and PV-DBOW. For each model we report the feature set with the higher mean validation F1.`));
A.push(...table("The three text versions on linear regression, logistic regression, the perceptron and Naive Bayes (validation, mean over models and labels).",
  [1500, 1050, 1050, 1200],
  ["Text features", "Mean F1", "Mean ROC-AUC", "Best toxicity F1"],
  ["word2vec", "doc2vec_dbow", "doc2vec"].map((f) => [`${FEAT[f]}${f === "word2vec" ? " (mean)" : ""}`, F(FC[f].mean_f1), F(FC[f].mean_auc), F(FC[f].best_toxicity_f1)]),
  ["l", "r", "r", "r"]));
A.push(h2("2.2", "Training and parameter choice"));
A.push(body(`**Parameter choice.** Every parameter is chosen on the validation split by F1 on that label, except where a model section says otherwise. The test split is used once, for the final numbers. Where a search on 1.78 million rows was too slow, it ran on a stratified sample (the same share of positives as the full data), and the best setting was refitted on all rows when the model allowed it. The number of training rows is given for every model.`));
A.push(body(`**Class imbalance.** Positives are rare: 8% for toxicity and under 1% for four labels. Most models use balanced class weights, so a mistake on a positive costs as much in total as all mistakes on negatives (Lab 2). A comment is predicted positive at the model's default cut: a probability of 0.5, or a decision value of 0.`));
A.push(body(`**Metrics.** We record every Lecture 2 metric for every model, label and split: accuracy, error rate, precision, recall, specificity, false-positive rate (FPR), F1 and ROC-AUC. Accuracy is misleading on this data: answering "not toxic" for every comment is already 92% accurate. We compare models by F1, which balances precision and recall at the chosen cut, and by ROC-AUC, which measures how well the model ranks toxic comments above clean ones at every cut. A moderation queue uses both: the ranking decides what a moderator reads first, and the cut decides what is flagged.`));
A.push(body(`**Hardware.** The light models ran on a laptop with 7.4 GB of RAM, reading the features in chunks from disk. The heavy models ran on a 16 GB laptop. All code uses scikit-learn [7].`));

// ---------------------------------------------------------------- 3 MODELS
A.push(h1(3, "The Models"));
A.push(body(`Table 2 summarises all models, ${Fg("all")} shows F1 and ROC-AUC for every model and label, and Table 3 gives every metric on the test split. This section describes each model in the order of the lectures.`));

const B = [];   // wide tables
B.push(...table("Every model on the toxicity label, validation split unless marked. Mean = mean over the six labels. Time is the mean fit time per label.",
  [1350, 820, 900, 2160, 560, 560, 560, 600, 600, 600, 560, 660],
  ["Model", "Text", "Train rows", "Chosen setting (toxicity)", "F1", "Prec.", "Recall", "AUC", "Mean F1", "Mean AUC", "Test F1", "Time"],
  ORDER.map((m) => [NAME[m], FEAT[M[m].best_features], N(rows(m)), SETTING(m), F(tox(m).f1), F(tox(m).precision), F(tox(m).recall),
    F(tox(m).roc_auc), F(mean(m).f1), F(mean(m).roc_auc), F(tox(m, "test").f1), mins(avgSec(m))]),
  ["l", "l", "r", "l", "r", "r", "r", "r", "r", "r", "r", "r"]));
B.push(...figure("all", "fig1_all_models.png", "Validation F1 (left) and ROC-AUC (right) of every model on every label, each model on its better text features. Models are sorted by mean F1.", 6.9));

function SETTING(m) {
  const p = p0(m);
  switch (m) {
    case "linear_regression": return `${p.class_weight} weights`;
    case "logistic_regression": return `C = ${p.C}, balanced`;
    case "perceptron": return `${p.passes} passes, balanced`;
    case "naive_bayes": return `${p.priors} priors`;
    case "decision_tree_id3": case "decision_tree_cart": return `depth ${p.max_depth}, ${N(p.nodes)} nodes`;
    case "decision_tree_c45": return `depth ${p.depth}, ccp_alpha ${Number(p.ccp_alpha).toExponential(1)}, ${N(p.nodes)} nodes`;
    case "random_forest": return `${p.n_estimators} trees, √features`;
    case "svm_linear": return `hinge, alpha ${p.alpha}`;
    case "svm_rbf": return `C = ${p.C}, gamma = scale`;
    case "mlp": return `hidden ${p.hidden_layer_sizes}`;
    case "adaboost": return `${p.n_estimators} stumps`;
    case "knn": return `k = ${p.n_neighbors}`;
    default: return "";
  }
}

// ---------------------------------------------------------------- 3.x one subsection per model
const flags = (m, t) => M[m].labels[t].validation.tp + M[m].labels[t].validation.fp;
const C = [];   // two-column text after the wide tables
let sec = 0;
function model(m, title, setup, choice, comment, extra = "") {
  if (!M[m]) return;
  sec += 1;
  C.push(h2(`3.${sec}`, title));
  C.push(body(`**Setup.** ${setup}`));
  C.push(body(`**Parameter choice.** ${choice}`));
  C.push(body(perf(m, extra)));
  C.push(body(`**Comment.** ${comment}`));
}

model("linear_regression", "Linear regression (Lecture 4)",
  `Fitted by least squares on the 0/1 label, not on the raw score, and a comment is positive when the output is at least 0.5. This is a linear probability model, used as a classifier. Solved with the normal equations, accumulated in chunks over all ${N(rows("linear_regression"))} rows.`,
  `No weights against balanced weights. Balanced weights won on ${count("linear_regression", (p) => p.class_weight === "balanced")} of 6 labels; without them, toxicity F1 is only ${F(p0("linear_regression").tried_val_f1.none)}, because few outputs reach 0.5 when 92% of the targets are 0.`,
  `A strong baseline: almost level with logistic regression in about ${mins(avgSec("linear_regression"))} per label. Its output is not a probability (it can fall below 0 or rise above 1), and the squared loss also punishes correct answers that are "too sure". It finds most toxic comments (recall ${F(tox("linear_regression").recall, 2)}) but flags many clean ones (precision ${F(tox("linear_regression").precision, 2)}).`);

model("logistic_regression", "Logistic regression (Lecture 5)",
  `Sigmoid output trained by maximum likelihood (L-BFGS), L2 regularisation, balanced class weights, all ${N(rows("logistic_regression"))} rows.`,
  `The inverse regularisation strength C from {0.01, 0.1, 1, 10, 100}, chosen on a 200,000-row stratified sample and refitted on all rows. C = 0.01, the strongest regularisation, won on ${count("logistic_regression", (p) => p.C === 0.01)} of 6 labels.`,
  `The best linear model. It outputs a probability, so the cut can be moved without retraining, and its weights show which features push a comment towards toxic. It trains in about ${mins(avgSec("logistic_regression"))} per label. Strong regularisation winning means that more freedom does not help: the limit is the straight-line boundary, not the amount of data.`);

model("perceptron", "Perceptron (Lecture 6)",
  `Rosenblatt's update rule with balanced class weights, all rows, trained in chunks of 100,000 rows.`,
  `The number of passes over the data, up to 10, stopping after 2 passes without a better validation F1. Between ${Math.min(...TARGETS.map((t) => M.perceptron.params[t].passes))} and ${Math.max(...TARGETS.map((t) => M.perceptron.params[t].passes))} passes were kept.`,
  `Fast (about ${mins(avgSec("perceptron"))} per label) but the weakest linear model. It stops updating once a point is on the right side, so it finds no margin, and when the classes overlap, as here, its weights keep jumping: validation F1 went up and down from pass to pass. It gives no probability.`);

model("naive_bayes", "Naive Bayes (Lecture 11)",
  `Gaussian Naive Bayes: one normal distribution per feature and class, all rows.`,
  `Class priors learned from the data against uniform priors (0.5/0.5). Learned priors won on ${count("naive_bayes", (p) => p.priors === "learned")} of 6 labels.`,
  `The fastest model and a useful baseline. Its assumption that features are independent given the label is wrong here: the 100 text dimensions describe one comment together, and the count features are skewed rather than normal. On the rare labels it flags far too many comments (${N(flags("naive_bayes", "threat"))} for threat, which has ${N(M.naive_bayes.labels.threat.validation.tp + M.naive_bayes.labels.threat.validation.fn)} positives in validation), because multiplying 133 densities makes its probabilities extreme.`);

const treeNote = (m) => (M[m] && M[m].n_train.toxicity < PR.rows.train ? ` The final tree was fitted on a ${N(M[m].n_train.toxicity)}-row stratified sample, because the full data did not fit in memory.` : "");
model("decision_tree_id3", "Decision tree: ID3 (Lecture 9)",
  `Splits chosen by information gain (entropy), no pruning, balanced class weights. scikit-learn has no true ID3: it splits a continuous feature at a threshold, which ID3 itself cannot do without first binning the feature (Lecture 10).${treeNote("decision_tree_id3")}`,
  `Maximum depth from {5, 10, 15, 20}, chosen on a 200,000-row sample.${M.decision_tree_id3 ? ` Depth ${p0("decision_tree_id3").max_depth} won for toxicity (${N(p0("decision_tree_id3").nodes)} nodes).` : ""}`,
  ID3_COMMENT());

model("decision_tree_cart", "Decision tree: CART (Lecture 10)",
  `Binary splits chosen by Gini impurity, balanced class weights.${treeNote("decision_tree_cart")}`,
  `Maximum depth from {5, 10, 15, 20}, chosen on a 200,000-row sample. Depth ${p0("decision_tree_cart").max_depth} won for toxicity (${N(p0("decision_tree_cart").nodes)} nodes).`,
  CART_COMMENT());

model("decision_tree_c45", "Decision tree: C4.5 (Lecture 10)",
  `Entropy splits, maximum depth 20, balanced class weights, and pruning chosen on the validation split; final tree on all rows. scikit-learn has no gain ratio or rule post-pruning, so cost-complexity pruning stands in for C4.5's pruning.`,
  `The pruning strength ccp_alpha: up to 20 values from the tree's pruning path, chosen by validation accuracy on a 300,000-row sample.`,
  `Easy to read as rules, but weak here. Each split looks at one feature, while the text signal is spread over 100 dimensions that only mean something together. Choosing the pruning by accuracy favoured almost no pruning (accuracy barely changes on an imbalanced label), so the toxicity tree has ${N(p0("decision_tree_c45").nodes)} nodes and fits its training data far better (F1 ${F(M.decision_tree_c45.labels.toxicity.train.f1)}) than new comments.`);

model("random_forest", "Random forest (Lecture 10)",
  `Bagged, fully grown trees with √133 features tried per split and class weights balanced inside each bootstrap sample, on a ${N(rows("random_forest"))}-row stratified sample.`,
  `The number of trees from {100, 200, 300}: ${count("random_forest", (p) => p.n_estimators === 100)} labels chose 100 trees and ${count("random_forest", (p) => p.n_estimators === 200)} chose 200.`,
  `Its ROC-AUC (${F(mean("random_forest").roc_auc)} on average) shows that it ranks comments sensibly, but at the 0.5 cut it almost never says "toxic": it flags ${N(flags("random_forest", "toxicity"))} of ${N(PR.rows.validation)} validation comments for toxicity, and at most ${Math.max(...["obscene", "threat", "sexual_explicit"].map((t) => flags("random_forest", t)))} for obscene, threat and sexual_explicit. Fully grown trees memorise their bootstrap sample (training F1 ${F(M.random_forest.labels.toxicity.train.f1, 2)}), so their votes on a new rare positive rarely reach a majority, a known weakness of forests on imbalanced data [15]. A lower cut would recover much of this. Slow to train and hard to interpret.`);

model("svm_linear", "Linear SVM (Lecture 12)",
  `Hinge loss with L2 regularisation, trained by stochastic gradient descent on all rows (an exact SVM solver is far too slow at 1.78 million rows), balanced class weights.`,
  `Fixed, not searched: regularisation alpha = ${p0("svm_linear").alpha}, scikit-learn's default.`,
  `Level with logistic regression, as expected: both draw a straight boundary and differ only in the loss. It gives a decision value instead of a probability and adds nothing over logistic regression here.`);

model("svm_rbf", "Kernel SVM with RBF kernel (Lecture 12)",
  `Soft-margin SVM with a Gaussian (RBF) kernel and balanced class weights, on a ${N(rows("svm_rbf"))}-row stratified sample (${P(rows("svm_rbf") / PR.rows.train)} of the training data). An exact kernel SVM costs at least the square of the number of rows to train [12], and one fit on ${N(rows("svm_rbf"))} rows already took ${mins(M.svm_rbf.seconds.toxicity)} for toxicity.`,
  `Fixed, not searched: C = ${p0("svm_rbf").C} and gamma = 1 / (features × feature variance), scikit-learn's "scale" setting. A grid search was too costly at this size; Section 4 measures the effect of the sample instead.`,
  `The best F1 of all models on ${leads("svm_rbf", "f1").length} of 6 labels (${leads("svm_rbf", "f1").map((t) => t.replace(/_/g, " ")).join(", ")}), although it saw only ${P(rows("svm_rbf") / PR.rows.train)} of the data and was not tuned. Its curved boundary fits the overlap between toxic and clean comments better than a straight line, and balanced weights put its default cut close to the best one (Section 4). The costs: training time grows quickly with rows, ${N(p0("svm_rbf").support_vectors)} support vectors must be stored and compared with every new comment, there is no probability, and it is hard to explain. Its training F1 (${F(M.svm_rbf.labels.toxicity.train.f1, 2)}) is far above validation, a sign that ${N(rows("svm_rbf"))} rows are too few for the boundary it draws.`);

model("mlp", "Neural network: MLP (Lecture 7)",
  `A feed-forward network with ReLU hidden layers and a sigmoid output, trained by back-propagation with the Adam optimiser and early stopping on 10% of the training rows, refitted on all rows. scikit-learn's MLP has no class weights, so it learns the real class balance.`,
  `Hidden layers from {(64), (128), (128, 64)}, chosen on a 300,000-row sample. Two layers (128, 64) won on ${count("mlp", (p) => p.hidden_layer_sizes === "(128, 64)")} labels, including toxicity; one layer won on the others.`,
  `The best ranking of all models: the highest ROC-AUC on ${leads("mlp", "roc_auc").length} of 6 labels. Without class weights it is cautious: ${P(tox("mlp").precision, 0)} of its "toxic" flags are right, but it finds only ${P(tox("mlp").recall, 0)} of toxic comments. Its output is a probability, so a lower cut trades precision for recall without retraining (Section 4). It trains in about ${mins(avgSec("mlp"))} per label on all rows, but its weights cannot be read as reasons.`);

model("adaboost", "AdaBoost (Lecture 13)",
  `100 decision stumps (one-split trees), boosted, on a ${N(rows("adaboost"))}-row stratified sample, no class weights.`,
  `Fixed: 100 stumps, the setting of the Lecture 13 example.`,
  `Good ranking (mean ROC-AUC ${F(mean("adaboost").roc_auc)}) but low recall at the default cut (${F(tox("adaboost").recall, 2)} for toxicity), because without class weights the boosted vote stays on the majority class. Each stump looks at one feature, which suits the text dimensions poorly. Training is sequential and took about ${mins(avgSec("adaboost"))} per label.`);

model("knn", "K-nearest neighbours (Lab 1, Assignment 1)",
  `Euclidean distance, majority vote, on a ${N(rows("knn"))}-row stratified sample, no class weights.`,
  `k from {1, 3, 5, 9, 15, 21}. k = 1 won on ${count("knn", (p) => p.n_neighbors === 1)} of 6 labels and k = ${p0("knn").n_neighbors} on toxicity: a larger k lets the majority class outvote the rare positives.`,
  `The weakest ranking (mean ROC-AUC ${F(mean("knn").roc_auc)}, close to chance on the rare labels). In 133 dimensions the distances from a comment to its nearest and its farthest neighbours become similar, so the nearest neighbours carry little signal [14]. Prediction is slow, since every new comment is compared with every stored one. Not a good fit.`);

B.push(...table("Every Lecture 2 metric on the test split, toxicity label (each model on its better text features). FPR = false-positive rate.",
  [1700, 1050, 1050, 1050, 1050, 1050, 1050, 1050, 1030],
  ["Model", "Accuracy", "Error rate", "Precision", "Recall", "Specificity", "FPR", "F1", "ROC-AUC"],
  ORDER.map((m) => { const t = tox(m, "test"); return [NAME[m], F(t.accuracy), F(t.error_rate), F(t.precision), F(t.recall), F(t.specificity), F(t.fpr), F(t.f1), F(t.roc_auc)]; }),
  ["l", "r", "r", "r", "r", "r", "r", "r", "r"]));

sec += 1;
C.push(h2(`3.${sec}`, "Bayesian network (Lecture 11): not run"));
C.push(body(`A Bayesian network needs a known structure, built from causal relationships or domain knowledge, and it stores a probability table for each node. Our 133 features do not fit this. The text dimensions and the scaled statistics are continuous, and embedding dimensions have no causal meaning, so no structure can be drawn from domain knowledge. Learning a structure without that knowledge depends heavily on the order of the features, and the probability tables would first require binning every feature, which discards information. Naive Bayes, the special case whose structure treats every feature as independent given the label, was run instead and represents this family.`));

// ---------------------------------------------------------------- 4 TOP 3, 5 CHOICE, 6 CONCLUSION
const D = [], WIDE = [];
TOP3_SECTION(D);
CHOICE_SECTION(D);

D.push(plainHead("AI USAGE DISCLOSURE"));
D.push(body(`This report was completed with assistance from Claude (Anthropic). The AI was used to write analysis, and feature-generation scripts, to derive the statistics and draw the figures from their outputs. All numbers in the report are read from the outputs of the submitted scripts, the decisions were reviewed by the authors, and the authors are accountable for the accuracy of the content.`));

D.push(plainHead("REFERENCES"));
const refs = [
  "Moaz Allam and Kareem Elhenawy. 2026. Detecting Toxic Comments in Online Discussions: A Literature Review and Dataset Survey. CSCE 3602 / DSCI 3415 class project, Phase 1 report. The American University in Cairo.",
  "Moaz Allam and Kareem Elhenawy. 2026. Detecting Toxic Comments in Online Discussions: Preparing the Civil Comments Data. CSCE 3602 / DSCI 3415 class project, Phase 2 report. The American University in Cairo.",
  "Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. In Companion Proceedings of The 2019 World Wide Web Conference. arXiv:1903.04561",
  "Quoc Le and Tomas Mikolov. 2014. Distributed Representations of Sentences and Documents. In Proceedings of the 31st International Conference on Machine Learning (ICML). PMLR 32(2), 1188-1196.",
  "Jey Han Lau and Timothy Baldwin. 2016. An Empirical Evaluation of doc2vec with Practical Insights into Document Embedding Generation. In Proceedings of the 1st Workshop on Representation Learning for NLP. ACL, 78-86.",
  "Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013. Efficient Estimation of Word Representations in Vector Space. arXiv:1301.3781",
  "Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2825-2830.",
  "Radim Řehůřek and Petr Sojka. 2010. Software Framework for Topic Modelling with Large Corpora. In Proceedings of the LREC 2010 Workshop on New Challenges for NLP Frameworks. ELRA, 45-50.",
  "Zachary C. Lipton, Charles Elkan, and Balakrishnan Naryanaswamy. 2014. Optimal Thresholding of Classifiers to Maximize F1 Measure. In Machine Learning and Knowledge Discovery in Databases (ECML PKDD 2014). Springer, 225-239.",
  "Xavier Bouthillier et al. 2021. Accounting for Variance in Machine Learning Benchmarks. In Proceedings of Machine Learning and Systems 3 (MLSys 2021). arXiv:2103.03098",
  "Thomas G. Dietterich. 1998. Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation 10, 7, 1895-1923.",
  "Léon Bottou and Chih-Jen Lin. 2007. Support Vector Machine Solvers. In Large-Scale Kernel Machines, L. Bottou, O. Chapelle, D. DeCoste, and J. Weston (Eds.). MIT Press, 1-27.",
  "Takaya Saito and Marc Rehmsmeier. 2015. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 10, 3, e0118432.",
  "Kevin Beyer, Jonathan Goldstein, Raghu Ramakrishnan, and Uri Shaft. 1999. When Is \"Nearest Neighbor\" Meaningful? In Database Theory (ICDT'99). LNCS 1540, Springer, 217-235.",
  "Chao Chen, Andy Liaw, and Leo Breiman. 2004. Using Random Forest to Learn Imbalanced Data. Technical Report 666, Department of Statistics, University of California, Berkeley.",
];
refs.forEach((r, i) => D.push(new Paragraph({ spacing: { after: 16, line: 176 }, indent: { left: 230, hanging: 230 },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, size: TINY })] })));

// ================================================================== DOCUMENT
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } };
const two = { count: 2, space: 475, equalWidth: true };
const doc = new Document({
  creator: "Moaz Allam, Kareem Elhenawy",
  title: "Detecting Toxic Comments: A Pilot Comparison of Supervised Models",
  styles: { default: { document: { run: { font: "Times New Roman", size: BODY } } } },
  sections: [
    { properties: { page }, children: title },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: A },
    { properties: { page, type: SectionType.CONTINUOUS }, children: B },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: [...C, ...D.slice(0, D.indexOf("SPLIT"))] },
    { properties: { page, type: SectionType.CONTINUOUS }, children: WIDE },
    { properties: { page, type: SectionType.CONTINUOUS, column: two }, children: D.slice(D.indexOf("SPLIT") + 1) },
  ],
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(DEST, buf);
  console.log("wrote", DEST, buf.length, "bytes;", figNo, "figures,", tabNo, "tables");
});

// ================================================================== text that depends on later results
function ABSTRACT() {
  const cut = NUM.subgroups.cut_sweep;
  return `We compare the supervised models of the course on the Civil Comments toxicity data: linear and logistic regression, the perceptron, Naive Bayes, three decision trees (ID3, CART, C4.5), a random forest, linear and RBF-kernel support vector machines (SVMs), a multi-layer perceptron (MLP), AdaBoost and k-nearest neighbours. Each predicts six toxicity attributes from ${PR.n_tabular} tabular features and a ${DS.vector_size}-dimension text embedding, either averaged Word2Vec or Doc2Vec; PV-DBOW Doc2Vec matches Word2Vec and beats PV-DM. Every setting is chosen on validation and reported on test with all Lecture 2 metrics. The RBF SVM has the best F1 at its default cut (toxicity ${F(tox("svm_rbf", "test").f1)}), and the MLP ranks comments best (ROC-AUC ${F(tox("mlp", "test").roc_auc)}). A deeper study of the top three (RBF SVM, MLP, logistic regression) with cross-validation, repeated seeds, label thresholds of 0.4 and 0.6, decision-cut sweeps and identity-group bias metrics shows that most of the difference at the default cut comes from how many comments each model flags: with a cut chosen on validation, the MLP reaches test F1 ${F(cut.mlp.test_at_best_cut.f1)} against ${F(cut.svm_rbf.test_at_best_cut.f1)} for the SVM. Results on the four rare labels are noisy for every model, and all three models rank clean comments that mention some identity groups too high. We choose the MLP for the rest of the project.`;
}
function ID3_COMMENT() {
  const i = "decision_tree_id3";
  return `Depth 20, the deepest setting, won on ${count(i, (p) => p.max_depth === 20)} of 6 labels. Its F1 is close to the other trees (mean ${F(mean(i).f1)}, against ${F(mean("decision_tree_cart").f1)} for CART), so entropy and Gini lead to much the same splits, but without pruning it ranks comments worst of the three trees (mean ROC-AUC ${F(mean(i).roc_auc)}). The tree fits its training rows far better than new ones (toxicity F1 ${F(M[i].labels.toxicity.train.f1)} on training, ${F(tox(i).f1)} on validation). Its rules are readable in principle, but a tree of ${N(p0(i).nodes)} nodes over embedding dimensions is not something a moderator can read.`;
}
function CART_COMMENT() {
  const c = "decision_tree_cart", deepest = count(c, (p) => p.max_depth === 20);
  return `Validation F1 rose with every extra level, and depth 20, the deepest setting tried, won on ${deepest} of 6 labels, so the tree is still short of what the data could support. It is slightly better than C4.5 (mean F1 ${F(mean(c).f1)} against ${F(mean("decision_tree_c45").f1)}) while trained on fewer rows, but it has the same limits: one feature per split, a ranking well below the linear models (mean ROC-AUC ${F(mean(c).roc_auc)}), and a large gap between training and validation F1 on toxicity (${F(M[c].labels.toxicity.train.f1)} against ${F(tox(c).f1)}).`;
}
function TOP3_SECTION(D) {
  const CV = NUM.cv, SD = NUM.seeds, TH = NUM.thresholds.results, SG = NUM.subgroups, TS = NUM.tests;
  const cv = (m, t, k = "f1") => CV.models[m][t][k];
  const sd = (m, t, k = "f1") => SD[m][t][k];
  const th = (m, c, k = "f1", s = "validation") => TH[m][c][s][k];
  const cut = (m) => SG.cut_sweep[m];
  const ids = Object.keys(SG.identities), I = (n, m) => SG.identities[n][m];
  const nm = (m) => NAME[m];
  const pm = (d, k = 3) => `${F(d.mean, k)} ± ${F(d.std, k)}`;
  const lab = (t) => t.replace(/_/g, " ");
  const bestCount = (key) => Object.fromEntries(TOP3.map((m) => [m, ids.filter((n) => TOP3.reduce((b, x) => (I(n, x)[key] > I(n, b)[key] ? x : b)) === m).length]));
  const subBest = bestCount("subgroup_auc"), bpsnBest = bestCount("bpsn_auc"), bnspBest = bestCount("bnsp_auc");
  const lowest = (m, key) => ids.reduce((b, n) => (I(n, m)[key] < I(b, m)[key] ? n : b));
  const toxRate = (n) => I(n, "mlp").positives / I(n, "mlp").n;
  const testTox = (tox("mlp", "test").tp + tox("mlp", "test").fn) / PR.rows.test;
  const RAW = (m, f, t) => NUM.raw_f1[m][f][t];
  const mc = TS.mcnemar, ci = TS.ci95;
  const posVal = (t) => M.logistic_regression.labels[t].validation.tp + M.logistic_regression.labels[t].validation.fn;

  D.push(h1(4, "The Top Three Models"));
  D.push(h2("4.1", "Choosing the three"));
  D.push(body(`We rank the models by both headline metrics, averaged over the six labels (Table 2). The RBF SVM has the highest mean F1 (${F(mean("svm_rbf").f1)}), the MLP the second-highest F1 (${F(mean("mlp").f1)}) and the highest mean ROC-AUC (${F(mean("mlp").roc_auc)}). Third place is close: logistic regression, linear regression and the linear SVM are within ${F(Math.max(...["logistic_regression", "linear_regression", "svm_linear"].map((m) => tox(m).f1)) - Math.min(...["logistic_regression", "linear_regression", "svm_linear"].map((m) => tox(m).f1)), 3)} of each other on toxicity F1. We take logistic regression, which has the highest mean F1 of the three (${F(mean("logistic_regression").f1)}) and, unlike the other two, outputs a real probability. The three also represent three different families: a kernel method, a neural network and a linear model. The rest of this section uses each model on its better text features: PV-DBOW for the SVM and Word2Vec for the other two.`));

  D.push(h2("4.2", "Head-to-head on the test split"));
  D.push(body(`Table 4 gives test F1 and ROC-AUC for every label. At the default cut the SVM has the best toxicity F1 and the MLP the best ROC-AUC. To check that these gaps are not luck of the test sample, we redrew the ${N(PR.rows.test)} test comments ${N(TS.bootstrap_resamples)} times with replacement and recomputed both metrics (a bootstrap). The 95% intervals for toxicity F1 are ${F(ci.svm_rbf.f1.low)}–${F(ci.svm_rbf.f1.high)} for the SVM, ${F(ci.mlp.f1.low)}–${F(ci.mlp.f1.high)} for the MLP and ${F(ci.logistic_regression.f1.low)}–${F(ci.logistic_regression.f1.high)} for logistic regression, and for ROC-AUC ${F(ci.mlp.roc_auc.low)}–${F(ci.mlp.roc_auc.high)} (MLP), ${F(ci.svm_rbf.roc_auc.low)}–${F(ci.svm_rbf.roc_auc.high)} (SVM) and ${F(ci.logistic_regression.roc_auc.low)}–${F(ci.logistic_regression.roc_auc.high)} (logistic regression). The F1 intervals do not overlap, and the MLP's ROC-AUC interval lies clear of the other two (those of the SVM and logistic regression touch), so the order is not an accident of the test sample.`));
  D.push(body(`Dietterich recommends McNemar's test for comparing two classifiers that can be trained only once [11]. It counts the comments that one model gets right and the other gets wrong. The MLP is right where the SVM is wrong on ${N(mc["svm_rbf vs mlp"].only_second_right)} comments and the reverse happens on ${N(mc["svm_rbf vs mlp"].only_first_right)} (χ² = ${N(mc["svm_rbf vs mlp"].chi2)}, far above 3.84, the 5% level). This measures errors of either kind, so it rewards the MLP for flagging fewer comments: it says the MLP is more often right, not that it finds more toxic comments. The two metrics we care about disagree, which is the first sign that the comparison depends on how the models are used.`));
  D.push(...table("Test F1 and ROC-AUC of the top three on every label, at each model's default cut.",
    [1350, 570, 570, 570, 570, 570, 570],
    ["Label", "SVM F1", "MLP F1", "LogReg F1", "SVM AUC", "MLP AUC", "LogReg AUC"],
    TARGETS.map((t) => [lab(t), ...["f1", "roc_auc"].flatMap((k) => TOP3.map((m) => F(M[m].labels[t].test[k])))]),
    ["l", "r", "r", "r", "r", "r", "r"]));

  D.push(h2("4.3", "Variability"));
  D.push(body(`A single number per model hides how much it would move if the experiment were repeated. Bouthillier et al. list the main sources of this variance: which rows are sampled for training and testing, the random initialisation, and the choice of settings, and find that the data sample matters most [10]. We measured each source that applies to our three models. Table 5 summarises the results; this section explains them.`));

  D.push(body(`**Data sample: cross-validation.** To compare the models on equal terms, we drew a stratified pool of ${N(CV.pool_rows)} training comments per label and ran ${CV.folds}-fold cross-validation, with the same folds for all three models, so each model trains on ${N(CV.train_rows_per_fold)} rows and is tested on the remaining fifth (${Fg("cv")}). On toxicity, F1 is ${pm(cv("svm_rbf", "toxicity"))} for the SVM, ${pm(cv("mlp", "toxicity"))} for the MLP and ${pm(cv("logistic_regression", "toxicity"))} for logistic regression (mean ± standard deviation over folds). Logistic regression barely moves from fold to fold: it has only 134 weights and a convex loss, so any large sample leads to almost the same model. The SVM and MLP move about three times as much.`));
  D.push(...figure("cv", "fig2_cv_spread.png", `F1 (top) and ROC-AUC (bottom) of the top three in ${CV.folds}-fold cross-validation on ${N(CV.pool_rows)} training comments per label. Dots are folds; bars are means.`));
  D.push(body(`The spread grows sharply on the rare labels. For threat, SVM F1 ranges from ${F(cv("svm_rbf", "threat").min)} to ${F(cv("svm_rbf", "threat").max)} across folds. The reason is the number of positives: the pool holds ${N(CV.pool_positives.threat)} threat comments, so each test fold has about ${N(CV.pool_positives.threat / CV.folds)}, and a few comments flagged correctly or not change F1 a lot. The same holds for ROC-AUC, which also needs positives to rank. The validation split has ${N(posVal("threat"))} threat positives, so its numbers are steadier, but F1 below 0.2 on the four rare labels should be read as a rough figure for every model.`));

  D.push(body(`**Training rows.** The cross-validation also isolates the effect of data size, since every model there saw ${N(CV.train_rows_per_fold)} rows. On toxicity, logistic regression scores the same with ${N(CV.train_rows_per_fold)} rows as with all ${N(PR.rows.train)} (F1 ${F(cv("logistic_regression", "toxicity").mean)} against ${F(tox("logistic_regression").f1)}): a straight boundary is already fully learned from a small sample. The MLP gains the most from data: its toxicity F1 rises from ${F(cv("mlp", "toxicity").mean)} to ${F(tox("mlp").f1)}, and its ROC-AUC from ${F(cv("mlp", "toxicity", "roc_auc").mean)} to ${F(tox("mlp").roc_auc)}. On ${N(CV.train_rows_per_fold)} rows it never flags a threat or sexual_explicit comment (F1 ${F(cv("mlp", "threat").mean)}), because without class weights the ${N(Math.round(CV.pool_positives.threat * 0.8))} or so positives it sees are outweighed by the negatives. At equal size the SVM has the best F1 on every label but sexual explicit and ranks toxicity about as well as the MLP (ROC-AUC ${F(cv("svm_rbf", "toxicity", "roc_auc").mean)} against ${F(cv("mlp", "toxicity", "roc_auc").mean)}). In other words, the MLP's lead in ranking comes from using all the data, which the SVM cannot do [12].`));

  D.push(body(`**Random seed and sample.** We retrained each model five times on toxicity and threat, evaluated on validation. The MLP was refitted on all rows with five different seeds, which change its starting weights and the 10% of rows held out for early stopping. The SVM was refitted on five different 50,000-row samples. Toxicity F1 is ${pm(sd("svm_rbf", "toxicity"))} for the SVM and ${pm(sd("mlp", "toxicity"))} for the MLP (range ${F(sd("mlp", "toxicity").min)}–${F(sd("mlp", "toxicity").max)}), while the MLP's ROC-AUC hardly moves (${pm(sd("mlp", "toxicity", "roc_auc"))}). The seed changes where the MLP's scores fall relative to the 0.5 cut more than how it orders comments, so F1 moves and ROC-AUC does not. On threat, F1 varies much more for both (SVM ${pm(sd("svm_rbf", "threat"))}, MLP ${pm(sd("mlp", "threat"))}). Logistic regression has a single optimum, so its seed has no effect, and it was not repeated.`));

  D.push(body(`**Text features.** The choice between Word2Vec and PV-DBOW matters little on the common labels and a lot on the rare ones. The SVM's toxicity F1 is ${F(RAW("svm_rbf", "doc2vec_dbow", "toxicity"))} on PV-DBOW and ${F(RAW("svm_rbf", "word2vec", "toxicity"))} on Word2Vec, but on identity attack the order flips (${F(RAW("svm_rbf", "doc2vec_dbow", "identity_attack"))} against ${F(RAW("svm_rbf", "word2vec", "identity_attack"))}), as it does on sexual explicit (${F(RAW("svm_rbf", "doc2vec_dbow", "sexual_explicit"))} against ${F(RAW("svm_rbf", "word2vec", "sexual_explicit"))}). A 50,000-row sample holds only about ${N(Math.round(pos.sexual_explicit * rows("svm_rbf")))} sexual explicit positives, and these labels are the noisiest in the cross-validation too (sexual explicit F1 ${F(cv("svm_rbf", "sexual_explicit").min)}–${F(cv("svm_rbf", "sexual_explicit").max)} across folds). One run per feature set cannot separate a real preference from this noise, so we keep one feature set per model.`));

  D.push(body(`**Label threshold.** Phase 1 committed to repeating the evaluation with the label defined at 0.4 and 0.6 instead of 0.5. We rebuilt the toxicity label at each threshold and retrained the three models (Table 5). A lower threshold makes positives more common (${P(TH.svm_rbf["0.4"].train_positive_rate)} of training comments at 0.4, ${P(TH.svm_rbf["0.6"].train_positive_rate)} at 0.6), and every model's F1 rises with it: the SVM's goes from ${F(th("svm_rbf", "0.6"))} at 0.6 to ${F(th("svm_rbf", "0.4"))} at 0.4. ROC-AUC moves the other way, because at 0.6 the positives are the comments most raters agreed on, which are easier to rank. The order of the three models is the same at every threshold, so the comparison does not depend on where the label line is drawn.`));

  D.push(body(`**Decision cut.** The largest source of variation is the cut that turns a score into a yes or no. ${Fg("cut")} shows validation F1 for every cut, expressed as the share of comments flagged. Each model's default cut sits at a different point on its curve. The SVM, trained with balanced weights, flags ${P((cut("svm_rbf").at_default.tp + cut("svm_rbf").at_default.fp) / PR.rows.validation)} of comments and is already near its best F1 (${F(cut("svm_rbf").at_default.f1)} against a best of ${F(cut("svm_rbf").best_on_validation.f1)}). Logistic regression, also balanced, flags ${P((cut("logistic_regression").at_default.tp + cut("logistic_regression").at_default.fp) / PR.rows.validation)}, too many. The MLP, trained without weights, flags only ${P((cut("mlp").at_default.tp + cut("mlp").at_default.fp) / PR.rows.validation)}, far too few. This analysis uses a refitted copy of each model, so its default-cut numbers differ slightly from Table 2. Choosing the cut on validation and applying it once to test gives test F1 ${F(cut("mlp").test_at_best_cut.f1)} for the MLP, ${F(cut("logistic_regression").test_at_best_cut.f1)} for logistic regression and ${F(cut("svm_rbf").test_at_best_cut.f1)} for the SVM. The order of Section 4.2 changes: with a tuned cut, the MLP is clearly the best classifier, as its higher ROC-AUC predicted.`));
  D.push(body(`The MLP's best cut, ${F(cut("mlp").best_on_validation.cut)}, is almost exactly half of its best F1 (${F(cut("mlp").best_on_validation.f1)}). Lipton et al. prove that this holds for a model whose scores are calibrated probabilities [9]. The MLP was trained on the real class balance, so its scores behave like probabilities; the balanced weights of logistic regression inflate its scores, which moves its best cut up to ${F(cut("logistic_regression").best_on_validation.cut)}. Saito and Rehmsmeier warn that on imbalanced data ROC-AUC can look good while precision is poor [13]; ${Fg("cut")} is the check on that, and it agrees with ROC-AUC here.`));
  D.push(...figure("cut", "fig3_cut_sweep.png", "Validation F1 on toxicity as the decision cut moves, shown as the share of comments flagged toxic (log scale). Dots mark each model's default cut: probability 0.5 for the MLP and logistic regression, decision value 0 for the SVM."));

  D.push(body(`**Identity groups.** Phase 1 also committed to checking whether the models treat comments about particular groups unfairly, with the metrics of Borkan et al. [3]. For each identity mentioned in at least 100 test comments (${ids.length} identities), we compute three ROC-AUCs on toxicity: inside the group (subgroup AUC); toxic comments outside the group against clean comments that mention it (BPSN, where a low value means clean comments about the group are scored as toxic); and the reverse (BNSP, where a low value means toxic comments about the group are missed). ${Fg("subgroups")} shows the results. The MLP has the highest subgroup AUC for all ${subBest.mlp === ids.length ? "" : `but ${ids.length - subBest.mlp} of the `}${ids.length} identities and the highest BNSP for ${bnspBest.mlp}; its BPSN is highest for ${bpsnBest.mlp} and the SVM's for the other ${bpsnBest.svm_rbf}.`));
  D.push(body(`All three models share the same weakness. Their BPSN is below their BNSP for nearly every group (${TOP3.map((m) => `${ids.filter((n) => I(n, m).bpsn_auc < I(n, m).bnsp_auc).length} of ${ids.length} for ${nm(m) === "Logistic regression" ? "logistic regression" : `the ${nm(m)}`}`).join(", ")}), and the lowest BPSN is always for the same group, "${lab(lowest("mlp", "bpsn_auc"))}" (${TOP3.map((m) => F(I(lowest(m, "bpsn_auc"), m).bpsn_auc)).join(", ")}). That is, a clean comment that mentions a group is ranked above toxic comments elsewhere too often. The cause is in the data: ${P(toxRate("white"))} of test comments that mention white people and ${P(toxRate("black"))} of those that mention black people are toxic, against ${P(testTox)} overall, so the identity words themselves become evidence of toxicity, exactly the bias Borkan et al. describe [3]. Logistic regression, which can only add up feature effects, shows it most (BPSN ${F(I("white", "logistic_regression").bpsn_auc)} for "white"). The MLP reduces the problem but does not remove it.`));
  D.push("SPLIT");
  WIDE.push(...figure("subgroups", "fig4_subgroups.png", `Bias metrics on the test split for the ${ids.length} identities mentioned in at least 100 test comments, toxicity label. Low BPSN: clean comments that mention the group are ranked as toxic; low BNSP: toxic comments that mention it are missed. Dotted lines mark each model's ROC-AUC on all test comments.`, 6.9));

  D.push(body(`**Validation against test.** Every parameter and cut was chosen on validation, so test is the honest check. Test F1 differs from validation F1 by at most ${F(Math.max(...TOP3.map((m) => Math.abs(tox(m, "test").f1 - tox(m).f1))))} on toxicity for the three models, and the tuned cuts carry over (MLP: ${F(cut("mlp").best_on_validation.f1)} on validation, ${F(cut("mlp").test_at_best_cut.f1)} on test). The large, well-mixed splits mean that choosing on validation did not over-fit it.`));

  D.push(...table("How much each result moves, toxicity label unless stated. ± is the standard deviation over folds or runs. The label-threshold and cut rows use retrained copies of each model, so they can differ slightly from Table 2.",
    [1500, 1100, 1100, 1100],
    ["Source", "RBF SVM", "MLP", "LogReg"],
    [
      ["CV folds (F1)", ...TOP3.map((m) => pm(cv(m, "toxicity")))],
      ["CV folds, threat (F1)", ...TOP3.map((m) => pm(cv(m, "threat")))],
      ["Seeds / samples (F1)", pm(sd("svm_rbf", "toxicity")), pm(sd("mlp", "toxicity")), "none (one optimum)"],
      ["Rows: 50k → all (F1)", "50k only", `${F(cv("mlp", "toxicity").mean)} → ${F(tox("mlp").f1)}`, `${F(cv("logistic_regression", "toxicity").mean)} → ${F(tox("logistic_regression").f1)}`],
      ["Label at 0.4 / 0.5 / 0.6 (F1)", ...TOP3.map((m) => ["0.4", "0.5", "0.6"].map((c) => F(th(m, c), 2)).join(" / "))],
      ["Default → best cut (test F1)", ...TOP3.map((m) => `${F(cut(m).test_at_default.f1)} → ${F(cut(m).test_at_best_cut.f1)}`)],
      ["Subgroup AUC (min–max)", ...TOP3.map((m) => `${F(Math.min(...ids.map((n) => I(n, m).subgroup_auc)), 2)}–${F(Math.max(...ids.map((n) => I(n, m).subgroup_auc)), 2)}`)],
      ["Validation → test (F1)", ...TOP3.map((m) => `${F(tox(m).f1)} → ${F(tox(m, "test").f1)}`)],
    ], ["l", "r", "r", "r"]));

  D.push(h2("4.4", "Why the results vary"));
  D.push(bullet(`• **Too few positives.** On the four rare labels, a fold or sample holds tens to a few hundred positives, so F1 and ROC-AUC are noisy for every model. This is a property of the data, not of any model.`));
  D.push(bullet(`• **Model capacity.** Logistic regression has few parameters and one optimum, so it is stable but cannot improve with more data. The SVM and MLP can draw curved boundaries, which helps but also lets them follow the particular rows they saw, so they vary more.`));
  D.push(bullet(`• **Training rows.** The SVM is limited to 50,000 rows by its cost; the MLP uses all 1.78 million. Comparisons at the default setup therefore mix model quality with data quantity.`));
  D.push(bullet(`• **Class weights and the cut.** Balanced weights shift scores up, no weights keep them low. Most of the F1 difference between the models at the default cut is a difference in how many comments each one flags, not in how well it ranks them.`));
  D.push(bullet(`• **Biased signal.** Identity words are more common in toxic comments, so all models over-flag clean comments that mention some groups.`));
}
function CHOICE_SECTION(D) {
  const cut = NUM.subgroups.cut_sweep, SG = NUM.subgroups, ids = Object.keys(SG.identities), CV = NUM.cv, SD = NUM.seeds;
  const testAucWins = TARGETS.filter((t) => TOP3.every((m) => m === "mlp" || M.mlp.labels[t].test.roc_auc >= M[m].labels[t].test.roc_auc)).length;
  D.push(h1(5, "Choice of Technique"));
  D.push(body(`We choose the **MLP** for the rest of the project, weighing three things.`));
  D.push(body(`**Performance.** The MLP ranks comments best: among the top three it has the highest test ROC-AUC on ${testAucWins} of 6 labels (Table 4) and the highest subgroup AUC for every one of the ${ids.length} identity groups. Its lower F1 at the 0.5 cut comes from flagging too few comments, not from ranking them badly. With a cut chosen on validation, its test F1 on toxicity is ${F(cut.mlp.test_at_best_cut.f1)}, well above the SVM (${F(cut.svm_rbf.test_at_best_cut.f1)}) and logistic regression (${F(cut.logistic_regression.test_at_best_cut.f1)}), and the cut carried over from validation to test unchanged (Section 4.3).`));
  D.push(body(`**Nature of the model.** The MLP learns curved boundaries, which the text embeddings need, and it is trained by gradient descent in small batches, so it scales to all ${N(PR.rows.train)} comments in about ${mins(avgSec("mlp"))} per label. The RBF SVM cannot: its cost grows at least with the square of the rows [12], so it was limited to ${N(rows("svm_rbf"))} rows, and at that size the MLP had lower F1 than the SVM on every label in cross-validation. The MLP's gain from more data (toxicity F1 ${F(CV.models.mlp.toxicity.f1.mean)} on ${N(CV.train_rows_per_fold)} rows, ${F(tox("mlp").f1)} on all) also suggests that it can still improve. Its scores behave like probabilities (its best cut is half its best F1 [9]), so a single cut has a clear meaning. Logistic regression has the same property and is more stable, but its straight boundary caps it, and it shows the strongest identity bias.`));
  D.push(body(`**Fit to the problem.** The moderation aid sorts comments so that a moderator reads the worst first, and flags those above a cut that the moderator can move to match the workload. That calls for a good ranking and a probability-like score, which is what the MLP provides. The task is also multi-label (Phase 1): one network with six sigmoid outputs could share its hidden layers between the attributes, which may help the rare labels.`));
  D.push(body(`**Weaknesses to address.** The MLP's F1 moves with its seed (toxicity ${F(SD.mlp.toxicity.f1.mean)} ± ${F(SD.mlp.toxicity.f1.std)}), it has no class weights, and its weights cannot be read as reasons. For the next phase we will choose the cut per label on validation, average several seeds, give rare positives more weight by oversampling them in training, and keep checking the identity-group metrics, since the MLP reduces the bias on mentions of groups such as "white" and "black" but does not remove it. Logistic regression stays as the interpretable baseline.`));

  D.push(h1(6, "Conclusion"));
  D.push(body(`We trained ${ORDER.length} supervised models from the course on the same ${N(PR.rows.train)} training comments and ${PR.n_tabular + DS.vector_size} features, and explained why the Bayesian network was not used. Models that combine the embedding dimensions (the SVMs, the MLP, linear and logistic regression) clearly beat models that split on one feature at a time (the trees, AdaBoost) or rely on distances (KNN). Comparing the top three in depth showed that the headline F1 depends as much on each model's default cut and on the number of rows it can use as on the model itself, and that rare labels give noisy results for every model. Measured fairly, the MLP is the strongest and most suitable technique, and it is our choice for the final phase.`));
}
