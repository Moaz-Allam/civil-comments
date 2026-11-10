/*
 * build_report.js -- writes the Phase 4 design report (ACM two-column layout, same page setup as
 * Phases 1-3). Every number is read from phase4/report/out/numbers.json (phase4/report/collect.py).
 *
 * Run:  python phase4/report/collect.py && python phase4/report/make_figures.py && node phase4/report/build_report.js
 */
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, ImageRun, Table, TableRow, TableCell,
  WidthType, AlignmentType, SectionType, BorderStyle, ShadingType, TableLayoutType,
} = require("docx");

const FIG = path.join(__dirname, "figures");
const DEST = path.join(__dirname, "..", "..", "Phase4_Design_Report.docx");
const NUM = JSON.parse(fs.readFileSync(path.join(__dirname, "out", "numbers.json"), "utf8"));
const META = NUM.meta, P3 = NUM.phase3;

// ------------------------------------------------------------------ number formatting
const N = (x) => Math.round(Number(x)).toLocaleString("en-US");
const F = (x, d = 3) => Number(x).toFixed(d);
const P = (frac, d = 1) => (100 * Number(frac)).toFixed(d) + "%";

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
const formula = (str) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 60, after: 60 },
  children: [new TextRun({ text: str, italics: true })] });
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

// ------------------------------------------------------------------ figures and tables
let figNo = 0;
const figRef = {};
const FIGS = ["network", "stage1", "focal"];
FIGS.forEach((k, i) => (figRef[k] = i + 1));
const Fg = (k) => `Figure ${figRef[k]}`;
function figure(key, file, caption, widthIn = 3.25) {
  figNo += 1;
  if (figRef[key] !== figNo) throw new Error(`figure "${key}" appears as ${figNo} but FIGS says ${figRef[key]}`);
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = buf.readUInt32BE(16), h = buf.readUInt32BE(20);
  const px = widthIn * 96;
  firstAfterHeading = true;
  const paras = [
    new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120 }, keepNext: true,
      children: [new ImageRun({ type: "png", data: buf, transformation: { width: px, height: px * h / w } })] }),
    new Paragraph({ spacing: { before: 55, after: 120 }, children: runs(`**Figure ${figNo}:** ${caption}`, { size: SMALL }) }),
  ];
  return widthIn > 3.5 ? [{ __wide: true, children: paras }] : paras;
}
let tabNo = 0;
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const rule = (sz = 4, color = "000000") => ({ style: BorderStyle.SINGLE, size: sz, color });
function table(caption, widths, header, rows, aligns) {
  tabNo += 1;
  const total = widths.reduce((a, b) => a + b, 0);
  const mk = (txt, i, isHead, isLast) => new TableCell({
    width: { size: widths[i], type: WidthType.DXA },
    borders: { top: isHead ? rule() : none, bottom: isHead ? rule() : isLast ? rule() : rule(2, "BFBFBF"), left: none, right: none },
    shading: isHead ? { type: ShadingType.CLEAR, color: "auto", fill: "EFEFEF" } : undefined,
    margins: { top: 40, bottom: 40, left: 70, right: 70 },
    children: [new Paragraph({ spacing: { line: 195 },
      alignment: (aligns && aligns[i] === "r") ? AlignmentType.RIGHT : AlignmentType.LEFT,
      children: runs(txt, { size: TINY, bold: isHead || undefined }) })],
  });
  const trs = [new TableRow({ tableHeader: true, cantSplit: true, children: header.map((h, i) => mk(h, i, true, false)) })];
  rows.forEach((r, k) => trs.push(new TableRow({ cantSplit: true, children: r.map((c, i) => mk(c, i, false, k === rows.length - 1)) })));
  firstAfterHeading = true;
  return [
    new Paragraph({ spacing: { before: 140, after: 60 }, keepNext: true, children: runs(`**Table ${tabNo}:** ${caption}`, { size: SMALL }) }),
    new Table({ width: { size: total, type: WidthType.DXA }, columnWidths: widths, layout: TableLayoutType.FIXED, rows: trs }),
    new Paragraph({ spacing: { after: 60 }, children: [] }),
  ];
}

// ------------------------------------------------------------------ derived numbers
const TARGETS = META.targets;
const LAB = (t) => t.replace(/_/g, " ");
const TEXT = { word2vec: "Word2Vec", doc2vec_dbow: "PV-DBOW", both: "both" };
const S1 = NUM.stage1, S2 = NUM.stage2, SEP = NUM.separate_summary, FIN = NUM.final, CMP = NUM.compare;
const lossName = (s) => s.loss === "focal" ? `focal (γ = ${s.gamma}, α = ${s.alpha})` : { bce: "BCE", wbce: "weighted BCE" }[s.loss];
const setName = (s) => `${lossName(s)}, layers ${s.hidden.join("-")}, ${TEXT[s.text]}`;
const find1 = (fn) => S1.find((r) => fn(r.setting));
const atBase = (r) => r.setting.hidden.join() === "256,128" && r.setting.text === "both";
const s1Best = S1[0];
const s1Bce = find1((s) => s.loss === "bce" && s.hidden.join() === "256,128" && s.text === "both");
const s1Wbce = find1((s) => s.loss === "wbce" && s.hidden.join() === "256,128" && s.text === "both");
const s1Focal = S1.filter((r) => r.setting.loss === "focal" && atBase(r));
const bestFocal = s1Focal[0];
const foldWins = (a, b) => Object.keys(a.fold_pr_auc).filter((f) => a.fold_pr_auc[f] > b.fold_pr_auc[f]).length;
const meanFold = (a, b) => Object.keys(a.fold_pr_auc).reduce((x, f) => x + a.fold_pr_auc[f] - b.fold_pr_auc[f], 0) / Object.keys(a.fold_pr_auc).length;
const p3mlp = P3.models.mlp, p3cut = P3.subgroups.cut_sweep;
const posRate = META.positive_rate;
const sci = (x) => { if (!x) return "0"; const e = Math.floor(Math.log10(x) + 1e-9), m = Math.round(x / 10 ** e);
  const sup = String(e).replace("-", "⁻").replace(/\d/g, (d) => "⁰¹²³⁴⁵⁶⁷⁸⁹"[d]); return `${m === 1 ? "" : m + "·"}10${sup}`; };
const actName = { relu: "ReLU", leaky_relu: "leaky ReLU", tanh: "tanh", sigmoid: "sigmoid" };
const s2IsBase = (s) => s.activation === "relu" && s.dropout === 0.2 && s.lr === 0.001 && s.batch === 2048 && s.weight_decay === 0 && s.identity_weight === 0;
const pending = (what) => `_{[${what}: the search is still running; this paragraph is filled in from its results.]}_`;

// ================================================================== CONTENT
const title = [
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 110 },
    children: [new TextRun({ text: "Detecting Toxic Comments in Online Discussions:", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 150 },
    children: [new TextRun({ text: "Model and Application Design", bold: true, size: 30 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 }, children: [new TextRun({ text: "Moaz Allam, Kareem Elhenawy", size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 },
    children: [new TextRun({ text: "Department of Computer Science and Engineering, The American University in Cairo", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 25 }, children: [new TextRun({ text: "Moaz_Allam@aucegypt.edu, karimelhenawy@aucegypt.edu", size: 19 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 },
    children: [new TextRun({ text: "CSCE 3602 / DSCI 3415 Class Project, Phase 4: Model and Application Design", size: 19, italics: true })] }),
];

const A = [];
A.push(new Paragraph({ spacing: { after: 50 }, children: [new TextRun({ text: "ABSTRACT", bold: true })] }));
firstAfterHeading = true;
A.push(body(ABSTRACT()));
A.push(new Paragraph({ spacing: { before: 100 }, alignment: AlignmentType.JUSTIFIED,
  children: [new TextRun({ text: "KEYWORDS  ", bold: true }), new TextRun({ text: "toxic comment classification, neural network, focal loss, cross-validation, hyperparameter search, client/server design" })] }));

// ---------------------------------------------------------------- 1 INTRODUCTION
A.push(h1(1, "Introduction"));
A.push(body(`Our project builds a moderation aid: it reads a comment and scores it on six toxicity attributes, so that a moderator can read the worst comments first. Phase 3 compared every supervised model of the course and chose a multi-layer perceptron (MLP) [3]. This phase designs the final model and the application that will use it.`));
A.push(body(`Section 2 explains how we answered the feedback on Phase 3. Section 3 restates why the MLP fits our data and our application. Section 4 designs the model: the data and validation scheme, the inputs, the network, the loss function, the hyperparameter search and its results, and the final model. Section 5 designs the application.`));

// ---------------------------------------------------------------- 2 FEEDBACK
A.push(h1(2, "Answering the Phase 3 Feedback"));
A.push(body(`Phase 3 received three comments. Table 1 lists each with what we changed.`));
A.push(...table("The Phase 3 feedback and our response.", [1500, 3300],
  ["Comment", "What we did"],
  [
    ["Validate with k-fold cross-validation, not a separate validation split.",
      `The validation split is merged into training, giving a pool of ${N(META.rows.pool)} comments. Every setting is scored by ${META.folds}-fold cross-validation on this pool, with folds grouped by article. Test (${N(META.rows.test)} comments) is used once, at the end (Section 4.1).`],
    ["The MLP choice is agreed.", "We keep the MLP and build it in PyTorch, so the loss and the layers can be changed (Section 3)."],
    ["Plan how to choose the hyperparameters; use focal loss rather than BCE; vary the activation, the hidden layer size, etc.",
      "A two-stage search over the loss (BCE, weighted BCE, focal with several γ and α), the layers, the text features, the activation, dropout, L2, learning rate, batch size and identity weights, on all rows (Sections 4.4 to 4.6)."],
  ]));

// ---------------------------------------------------------------- 3 ALGORITHM
A.push(h1(3, "Choice of Algorithm"));
A.push(body(`We keep the MLP chosen in Phase 3. It ranked comments best of the top three models (test ROC-AUC ${F(p3mlp.labels.toxicity.test.roc_auc)} on toxicity), and with a cut chosen on held-out data it reached test F1 ${F(p3cut.mlp.test_at_best_cut.f1)}, against ${F(p3cut.svm_rbf.test_at_best_cut.f1)} for the RBF SVM and ${F(p3cut.logistic_regression.test_at_best_cut.f1)} for logistic regression [3].`));
A.push(body(`It also fits the data and the application better than the other two. It trains by gradient descent in small batches, so it uses all ${N(p3mlp.n_train.toxicity)} training comments, while the RBF SVM was limited to ${N(P3.models.svm_rbf.n_train.toxicity)} rows by its cost. Phase 3 showed that the MLP is the model that gains most from data. It outputs a probability-like score, so a moderator can sort a queue by it and move the cut to match the workload. Prediction is one small matrix product per layer, fast enough for a server that scores comments as they arrive. Its weaknesses are that its weights cannot be read as reasons, and that its results move a little with the random seed; Section 4.7 reduces the second by averaging five seeds.`));

// ---------------------------------------------------------------- 4 MODEL DESIGN
A.push(h1(4, "Model Design"));
A.push(h2("4.1", "Data and validation"));
A.push(body(`**Pool and test.** The ${N(META.rows.pool)} training and validation comments form one pool; the ${N(META.rows.test)} test comments are kept apart and used once. The six labels are positive when at least half the raters said so, as in Phase 1; in the pool, ${P(posRate.toxicity)} of comments are toxic and between ${P(posRate.threat, 2)} and ${P(posRate.identity_attack, 2)} carry each of the four rare labels.`));
A.push(body(`**Folds.** Comments under the same news article share topic, words and quoted text, and Phase 2 found that the official splits keep articles apart. Our folds do the same: each of the ${N(META.articles)} articles goes to exactly one fold (scikit-learn's StratifiedGroupKFold). StratifiedGroupKFold balances one class label, but we have six. We give each comment the rarest label it carries as its stratification key, so every fold receives a fair share of the rare labels. Table 2 shows the result: each fold holds ${N(Math.min(...META.fold_positives.threat))} to ${N(Math.max(...META.fold_positives.threat))} threat comments, the rarest label.`));
A.push(...table(`Positives per fold (pool of ${N(META.rows.pool)} comments).`, [1300, 700, 700, 700, 700, 700],
  ["Label", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Fold 5"],
  TARGETS.map((t) => [LAB(t), ...META.fold_positives[t].map(N)]), ["l", "r", "r", "r", "r", "r"]));
A.push(body(`**Selection metric.** Each setting is judged by its mean PR-AUC (average precision) over the six labels, computed on out-of-fold scores. PR-AUC measures how well positives are ranked above negatives while paying attention to precision, which ROC-AUC can hide when positives are rare [10]. It does not depend on a cut, so the cut is chosen afterwards (Section 4.7).`));

A.push(h2("4.2", "Inputs"));
A.push(body(`Each comment is described by ${META.columns.tabular[1]} tabular features from Phase 2 (text statistics, a reply flag and the publication) and two 100-dimension text vectors: the mean of the comment's Word2Vec word vectors and its PV-DBOW Doc2Vec vector. Which text vectors the network sees is a search choice: Word2Vec, PV-DBOW, or both (233 inputs).`));
A.push(body(`**No manual feature weighting.** Every input is already standardised with the training mean and standard deviation (Phase 2), so no feature dominates by its scale. The first layer then learns one weight per input and hidden unit, which is a learned feature weighting; fixing weights by hand would only constrain it. The search does decide how much of the text the network sees, by choosing between the text vectors.`));

A.push(h2("4.3", "Network"));
A.push(body(`${Fg("network")} shows the design. The inputs pass through one to three fully connected hidden layers, each followed by an activation function and dropout, and end in six sigmoid outputs, one per label. In the search, one network predicts all six labels, so the rare labels share the hidden layers learned from the common ones; Section 4.6 compares this with six separate single-label networks of the same design, which do better and form the final model. Each output's bias starts at the log-odds of its label's positive rate, so the untrained network already predicts the base rate; Lin et al. found this keeps training stable when positives are rare [5]. Training uses the Adam optimiser in mini-batches and stops early when the mean PR-AUC on a random ${P(0.05, 0)} of the training rows has not improved for 3 epochs (at most 30 epochs).`));
A.push(...figure("network", "fig1_network.png", "The model. The search chooses the text vectors, the hidden layers and their activation and dropout, and the loss; each label's cut is chosen from out-of-fold scores.", 6.9));

A.push(h2("4.4", "Loss function"));
A.push(body(`For a comment with label y ∈ {0, 1} and predicted probability p, let p_t = p when y = 1 and 1 − p when y = 0. The search compares three losses, averaged over the six labels:`));
A.push(bullet(`• **Binary cross-entropy (BCE):** −log p_t. The standard loss, used by the Phase 3 MLP.`));
A.push(bullet(`• **Weighted BCE:** positives weighted by (number of negatives / number of positives) for each label, the "balanced" weighting used for most models in Phase 3.`));
A.push(bullet(`• **Focal loss** [5]: −α_t (1 − p_t)^γ log p_t, where α_t = α for positives and 1 − α for negatives. With γ = 0 and α = 0.5 it is BCE up to a constant. A larger γ shrinks the loss of comments the model already gets right, so training concentrates on hard ones, which is why the course feedback recommended it for scarce labels. Lin et al. found γ = 2 and α = 0.25 best for object detection [5]. We try γ ∈ {0.5, 1, 2, 3} and α ∈ {0.25, 0.5, balanced}, where balanced sets α to 1 minus the label's positive rate.`));
A.push(body(`For γ < 1 the factor (1 − p_t)^γ has an infinite slope when p_t reaches exactly 1, which made training diverge in our first run; we clamp 1 − p_t at 10⁻⁶, which changes nothing for γ ≥ 1.`));
A.push(body(`**Identity weights.** Phase 3 found that all models rank clean comments that mention groups such as "white" or "black" too high [3]. Following Zhang et al., who reduce this bias by weighting training comments [12], a comment that mentions an identity (an identity score of at least 0.5) can count (1 + w) times in the loss, with w ∈ {0, 1, 3} as a search choice; w = 0 is no weighting.`));

A.push(h2("4.5", "Hyperparameter search"));
A.push(body(`Random search finds good settings with far fewer trials than a grid, because only a few hyperparameters matter much [1]. We search in two stages, each setting scored by ${META.folds}-fold cross-validation on all pool rows (Table 3). Stage 1 fixes the "shape" of the model: the loss, the hidden layers and the text vectors, sampled at random, plus BCE, weighted BCE and focal (γ = 2, α = 0.25) always at the base setting, and a focal grid of γ × α at the base setting so that focal loss gets a fair test against BCE. Stage 2 tunes the training around the stage-1 winner: activation, dropout, learning rate, batch size and L2, sampled at random, plus the identity weights.`));
A.push(...table("The search space. The base setting is in bold.", [1350, 3450],
  ["Hyperparameter", "Values"],
  [
    ["Loss (stage 1)", "**BCE**, weighted BCE, focal γ ∈ {0.5, 1, 2, 3} × α ∈ {0.25, 0.5, balanced}"],
    ["Hidden layers (stage 1)", "256; 512; **256-128**; 512-256; 512-256-128"],
    ["Text vectors (stage 1)", "Word2Vec; PV-DBOW; **both**"],
    ["Activation (stage 2)", "**ReLU**, leaky ReLU, tanh, sigmoid"],
    ["Dropout (stage 2)", "0, 0.1, **0.2**, 0.3"],
    ["Learning rate (stage 2)", "3·10⁻⁴, **10⁻³**, 3·10⁻³"],
    ["Batch size (stage 2)", "1,024, **2,048**, 4,096"],
    ["L2 weight decay (stage 2)", "**0**, 10⁻⁵, 10⁻⁴"],
    ["Identity weight w (stage 2)", "**0**, 1, 3"],
  ]));
A.push(body(`The search ran on a Google Colab GPU while one was available and on a laptop CPU afterwards; every finished fold is saved, so an interrupted run resumes where it stopped.`));

A.push(h2("4.6", "Search results"));
STAGE1_TEXT().forEach((s) => A.push(body(s)));
A.push(...figure("stage1", "fig2_stage1.png", `Stage 1: mean PR-AUC of every setting over ${META.folds} folds (bars: ± one standard deviation over folds).`, 6.9));
if (fs.existsSync(path.join(FIG, "fig3_focal.png")))
  A.push(...figure("focal", "fig3_focal.png", "Focal loss against BCE at the base setting (layers 256-128, both text vectors)."));
STAGE2_TEXT().forEach((s) => A.push(body(s)));
if (S2.length) A.push(...table(`Stage 2: every setting around the stage-1 winner (BCE, layers 256-128, both text vectors), ${META.folds}-fold CV. The base setting is in bold.`, [2700, 1050, 1050],
  ["Activation · dropout · lr · batch · L2 · w", "PR-AUC", "Epochs"],
  S2.map((r) => { const s = r.setting, isBase = s2IsBase(s);
    const name = `${actName[s.activation]} · ${s.dropout} · ${sci(s.lr)} · ${N(s.batch)} · ${sci(s.weight_decay)} · ${s.identity_weight}`;
    return [isBase ? `**${name}**` : name, `${F(r.mean_pr_auc)} ± ${F(r.std_pr_auc)}`, F(r.epochs, 1)]; }),
  ["left", "right", "right"]));

SEP_TEXT().forEach((s) => A.push(body(s)));

A.push(h2("4.7", "Final model and evaluation"));
FINAL_TEXT().forEach((s) => A.push(body(s)));
if (FIN && CMP) A.push({ __wide: true, children: table(`Final model on the test split (${N(META.rows.test)} comments, used once). Cut: max-F1 cut on out-of-fold scores. CI: 95% bootstrap interval (${N(CMP.resamples)} resamples of test rows).`,
  [1900, 800, 1100, 1100, 1000, 1500, 1100, 1500],
  ["Label", "Cut", "Precision", "Recall", "F1", "F1 95% CI", "PR-AUC", "PR-AUC 95% CI"],
  TARGETS.map((t) => { const r = FIN.test[t], c = CMP.labels[t];
    return [LAB(t), F(FIN.cuts[t]), F(r.precision), F(r.recall), F(r.f1), `${F(c.f1_ci[0])} – ${F(c.f1_ci[1])}`, F(r.pr_auc), `${F(c.pr_auc_ci[0])} – ${F(c.pr_auc_ci[1])}`]; }),
  ["l", "r", "r", "r", "r", "r", "r", "r"]) });
FINAL_TEXT2().forEach((s) => A.push(body(s)));

// ---------------------------------------------------------------- 5 APPLICATION
A.push(h1(5, "Application Design"));
A.push(body("_{[Application design: written by Kareem Elhenawy; to be added.]}_"));

// ---------------------------------------------------------------- 6 CONCLUSION
A.push(h1(6, "Conclusion"));
A.push(body(CONCLUSION()));

A.push(plainHead("AI USAGE DISCLOSURE"));
A.push(body(`This report was completed with assistance from Claude (Anthropic). The AI was used to write analysis, and feature-generation scripts, to derive the statistics and draw the figures from their outputs. All numbers in the report are read from the outputs of the submitted scripts, the decisions were reviewed by the authors, and the authors are accountable for the accuracy of the content.`));

A.push(plainHead("REFERENCES"));
const refs = [
  "James Bergstra and Yoshua Bengio. 2012. Random Search for Hyper-Parameter Optimization. Journal of Machine Learning Research 13, 281-305.",
  "Daniel Borkan, Lucas Dixon, Jeffrey Sorensen, Nithum Thain, and Lucy Vasserman. 2019. Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification. In Companion Proceedings of The 2019 World Wide Web Conference. arXiv:1903.04561",
  "Moaz Allam and Kareem Elhenawy. 2026. Detecting Toxic Comments in Online Discussions: A Pilot Comparison of Supervised Models. CSCE 3602 / DSCI 3415 class project, Phase 3 report. The American University in Cairo.",
  "Diederik P. Kingma and Jimmy Ba. 2015. Adam: A Method for Stochastic Optimization. In Proceedings of ICLR 2015. arXiv:1412.6980",
  "Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, and Piotr Dollár. 2017. Focal Loss for Dense Object Detection. In Proceedings of ICCV 2017, 2980-2988.",
  "Zachary C. Lipton, Charles Elkan, and Balakrishnan Naryanaswamy. 2014. Optimal Thresholding of Classifiers to Maximize F1 Measure. In ECML PKDD 2014. Springer, 225-239.",
  "Jishnu Mukhoti, Viveka Kulharia, Amartya Sanyal, Stuart Golodetz, Philip H. S. Torr, and Puneet K. Dokania. 2020. Calibrating Deep Neural Networks using Focal Loss. In Advances in Neural Information Processing Systems 33 (NeurIPS 2020).",
  "Adam Paszke et al. 2019. PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems 32 (NeurIPS 2019).",
  "Fabian Pedregosa et al. 2011. Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research 12, 2825-2830.",
  "Takaya Saito and Marc Rehmsmeier. 2015. The Precision-Recall Plot Is More Informative than the ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets. PLOS ONE 10, 3, e0118432.",
  "Nitish Srivastava, Geoffrey Hinton, Alex Krizhevsky, Ilya Sutskever, and Ruslan Salakhutdinov. 2014. Dropout: A Simple Way to Prevent Neural Networks from Overfitting. Journal of Machine Learning Research 15, 1929-1958.",
  "Guanhua Zhang, Bing Bai, Junqi Zhang, Kun Bai, Conghui Zhu, and Tiejun Zhao. 2020. Demographics Should Not Be the Reason of Toxicity: Mitigating Discrimination in Text Classifications with Instance Weighting. In Proceedings of ACL 2020, 4134-4145.",
];
refs.forEach((r, i) => A.push(new Paragraph({ spacing: { after: 16, line: 176 }, indent: { left: 230, hanging: 230 },
  children: [new TextRun({ text: `[${i + 1}] ${r}`, size: TINY })] })));

// ================================================================== DOCUMENT
const page = { size: { width: 12240, height: 15840 }, margin: { top: 1080, right: 1080, bottom: 1080, left: 1080 } };
const two = { count: 2, space: 475, equalWidth: true };
// wide figures get their own one-column section between two-column ones
const sections = [{ properties: { page }, children: title }];
let cur = [];
const flush = () => { if (cur.length) sections.push({ properties: { page, type: SectionType.CONTINUOUS, column: two }, children: cur }); cur = []; };
for (const item of A) {
  if (item && item.__wide) { flush(); sections.push({ properties: { page, type: SectionType.CONTINUOUS }, children: item.children }); }
  else cur.push(item);
}
flush();
const doc = new Document({
  creator: "Moaz Allam, Kareem Elhenawy",
  title: "Detecting Toxic Comments: Model and Application Design",
  styles: { default: { document: { run: { font: "Times New Roman", size: BODY } } } },
  sections,
});
Packer.toBuffer(doc).then((buf) => {
  fs.writeFileSync(DEST, buf);
  console.log("wrote", DEST, buf.length, "bytes;", figNo, "figures,", tabNo, "tables");
});

// ================================================================== text that depends on the search results
function ABSTRACT() {
  return `This report designs the final model and the application of our toxic-comment moderation aid. Following the Phase 3 comparison and its feedback, the model is a multi-layer perceptron built in PyTorch that predicts six toxicity attributes from ${META.columns.tabular[1]} tabular features and two text embeddings. Every design choice is made by ${META.folds}-fold cross-validation on ${N(META.rows.pool)} comments, with folds grouped by article and stratified on the rarest label, and judged by mean PR-AUC over the six labels. A two-stage search compares binary cross-entropy, weighted cross-entropy and focal loss, and tunes the layers, text features, activation, dropout, learning rate, batch size, L2 and identity weights. ${FIN && CMP ? `The search keeps BCE (focal loss is no better once the output bias starts at the label prior), ReLU and dropout 0.2, and six single-label networks beat one six-output network. Averaged over five seeds, the final model reaches test PR-AUC ${F(FIN.test.toxicity.pr_auc)} and F1 ${F(FIN.test.toxicity.f1)} on toxicity, against ${F(CMP.vs_phase3.phase3.pr_auc)} and ${F(CMP.vs_phase3.phase3.f1)} for the Phase 3 MLP, and improves the bias metrics of almost every identity group without removing the gap for the weakest ones. ` : "[Results pending.] "}The application is a client/server design: the model and its feature pipeline sit on a server behind an HTTP API, and a web client sends a comment and shows its six scores.`;
}

function STAGE1_TEXT() {
  if (!S1.length) return [pending("Stage 1 results")];
  const out = [];
  out.push(`Stage 1 finished ${S1.length} settings (${Fg("stage1")}). The best is ${setName(s1Best.setting)}, with mean PR-AUC ${F(s1Best.mean_pr_auc)} ± ${F(s1Best.std_pr_auc)} over the folds. Three patterns are clear. Both text vectors together beat either alone, and PV-DBOW alone is weakest: the two embeddings carry different information. Weighted BCE is clearly worse than plain BCE (${F(s1Wbce.mean_pr_auc)} against ${F(s1Bce.mean_pr_auc)} at the base setting, lower on all ${Object.keys(s1Bce.fold_pr_auc).length} folds): the weights reach ${N((1 - posRate.threat) / posRate.threat)} for threat positives, and such large weights hurt the ranking even though balanced weights helped the default cut of most Phase 3 models. And among focal settings, α = 0.25 beats the balanced α, as Lin et al. also found [5].`);
  if (bestFocal) {
    const d = meanFold(s1Bce, bestFocal);
    out.push(`The fair focal test at the base setting (${s1Focal.length} focal settings) gives a best focal loss of γ = ${bestFocal.setting.gamma}, α = ${bestFocal.setting.alpha} at ${F(bestFocal.mean_pr_auc)}, against ${F(s1Bce.mean_pr_auc)} for BCE; ${d > 0 ? `BCE is ahead by ${F(d, 4)} on average and on ${foldWins(s1Bce, bestFocal)} of ${Object.keys(s1Bce.fold_pr_auc).length} folds` : `focal is ahead by ${F(-d, 4)} on average and on ${foldWins(bestFocal, s1Bce)} of ${Object.keys(s1Bce.fold_pr_auc).length} folds`}. ${bestFocal.pr_auc.threat > s1Bce.pr_auc.threat ? `Focal loss is ahead on the rarest label, threat (PR-AUC ${F(bestFocal.pr_auc.threat)} against ${F(s1Bce.pr_auc.threat)}).` : ""} The gap is smaller than the fold-to-fold spread, so the two losses rank comments about equally well here; larger γ is clearly worse (${Fg("focal")}). A likely reason is that the output bias already starts at each label's prior, which removes most of the early flood of easy negatives that focal loss was designed against [5]. We keep BCE for stage 2 because it is the simpler loss and it is not worse.`);
  }
  return out;
}

function STAGE2_TEXT() {
  if (!S2.length) return [pending("Stage 2 results")];
  const base = S2.find((r) => s2IsBase(r.setting));
  const best = S2[0], worst = S2[S2.length - 1];
  const idw = (w) => S2.find((r) => r.setting.identity_weight === w);
  const bias = (r) => (NUM.stage2_summary || []).find((t) => JSON.stringify(t.setting) === JSON.stringify(r.setting)).bias;
  const meanOf = (b, k) => Object.values(b).reduce((x, v) => x + v[k], 0) / Object.keys(b).length;
  const minBpsn = (b) => Object.entries(b).sort((x, y) => x[1].bpsn_auc - y[1].bpsn_auc)[0];
  const tanh = S2.find((r) => r.setting.activation === "tanh");
  const sigLow = S2.find((r) => r.setting.activation === "sigmoid" && r.setting.lr < 1e-3);
  const out = [];
  out.push(`Stage 2 finished ${S2.length} settings (Table ${tabNo + 1}). ${best === base ? `None beats the base setting, which stays best at ${F(base.mean_pr_auc)} ± ${F(base.std_pr_auc)}` : `The best is ${actName[best.setting.activation]} at ${F(best.mean_pr_auc)}, against ${F(base.mean_pr_auc)} for the base setting`}. The training choices matter much less than the stage-1 choices: the best five settings, all ReLU or leaky ReLU with dropout 0.2 or 0.3, lie within ${F(S2[0].mean_pr_auc - S2[4].mean_pr_auc)} of each other. ReLU and leaky ReLU are the best activations; ${tanh ? `tanh (${F(tanh.mean_pr_auc)}) and ` : ""}sigmoid are worse, and sigmoid with the small learning rate ${sigLow ? `learns so slowly (small gradients) that after ${F(sigLow.epochs, 1)} epochs on average it reaches only ${F(sigLow.mean_pr_auc)}` : "is worst"}. Strong L2 (10⁻⁴) lowers PR-AUC in every setting that used it.`);
  if (idw(1) && idw(3) && NUM.stage2_summary) {
    const b0 = bias(base), b1 = bias(idw(1)), b3 = bias(idw(3));
    const [g0, v0] = minBpsn(b0), [g1, v1] = minBpsn(b1), [g3, v3] = minBpsn(b3);
    out.push(`The identity weights do not pay off. With w = 1 and w = 3, PR-AUC falls to ${F(idw(1).mean_pr_auc)} and ${F(idw(3).mean_pr_auc)}, and the bias metrics on the out-of-fold toxicity scores barely move: mean subgroup AUC ${F(meanOf(b0, "subgroup_auc"))}, ${F(meanOf(b1, "subgroup_auc"))} and ${F(meanOf(b3, "subgroup_auc"))}, and mean BPSN AUC ${F(meanOf(b0, "bpsn_auc"))}, ${F(meanOf(b1, "bpsn_auc"))} and ${F(meanOf(b3, "bpsn_auc"))} for w = 0, 1 and 3. The weakest group is "${g0}" in all three (BPSN ${F(v0.bpsn_auc)}, ${F(v1.bpsn_auc)} and ${F(v3.bpsn_auc)}), as in Phase 3 [3]. Weighting every identity comment the same does not fix the problem, because BPSN measures how often non-toxic comments that mention a group outrank toxic ones without it; up-weighting both toxic and non-toxic identity comments leaves that ordering largely unchanged. We therefore keep w = 0.`);
  }
  return out;
}

function SEP_TEXT() {
  if (!SEP || NUM.separate.length < TARGETS.length) return [pending("One network against six")];
  const joint = S2.find((r) => s2IsBase(r.setting)) || S2[0];
  const sep = Object.fromEntries(NUM.separate.map((r) => [r.labels[0], r]));
  const folds = Object.keys(joint.fold_pr_auc);
  const sepFold = (f) => TARGETS.reduce((x, t) => x + sep[t].fold_pr_auc[f], 0) / TARGETS.length;
  const wins = folds.filter((f) => sepFold(f) > joint.fold_pr_auc[f]).length;
  const better = TARGETS.filter((t) => sep[t].pr_auc[t] > joint.pr_auc[t]);
  const worse = TARGETS.filter((t) => sep[t].pr_auc[t] <= joint.pr_auc[t]);
  const ep = (ts) => ts.reduce((x, t) => x + sep[t].epochs, 0) / ts.length;
  const nIn = META.columns.tabular[1] - META.columns.tabular[0] + (joint.setting.text === "both" ? 200 : 100);
  const params = (outs) => { const h = [nIn, ...joint.setting.hidden, outs]; let n = 0; for (let i = 1; i < h.length; i++) n += h[i - 1] * h[i] + h[i]; return n; };
  const rare = ["identity_attack", "obscene", "threat", "sexual_explicit"].filter((t) => sep[t]);
  return [`Last, we trained six single-label networks with the best stage-2 setting, one per label, and compared them with the one six-output network. The six networks reach mean PR-AUC ${F(SEP.separate_mean_pr_auc)}, against ${F(SEP.joint_mean_pr_auc)} for the joint network, and they are ahead on ${wins} of ${folds.length} folds. They are better on ${better.map(LAB).join(", ")}${worse.length ? ` and slightly worse on ${worse.map(LAB).join(", ")} (${worse.map((t) => `${F(sep[t].pr_auc[t])} against ${F(joint.pr_auc[t])}`).join("; ")})` : ""}; the biggest gain is on ${LAB(better.sort((a, b) => (sep[b].pr_auc[b] - joint.pr_auc[b]) - (sep[a].pr_auc[a] - joint.pr_auc[a]))[0])}. The likely reason is early stopping: the joint network stops at one epoch for all six labels (${F(joint.epochs, 1)} epochs on average), which suits the common labels, while the single-label networks for the rare labels stop after only ${F(ep(rare), 1)} epochs on average, before they overfit their few positives. The final model therefore uses six networks. The cost is small: with ${nIn} inputs and layers ${joint.setting.hidden.join("-")}, the six networks have ${N(6 * params(1))} weights in total, against ${N(params(6))} for the joint network, still only about ${F(6 * params(1) * 4 / 1e6, 1)} MB as 32-bit floats.`];
}

function FINAL_TEXT() {
  const out = [`The final model is the best stage-2 setting${SEP ? (SEP.separate_mean_pr_auc > SEP.joint_mean_pr_auc ? " as six single-label networks" : " as one six-output network") : " (one network or six, as Section 4.6 decides)"}, trained five times with different seeds on the whole pool; its scores are the average of the five networks. Averaging removes most of the seed-to-seed variation that Phase 3 measured for the MLP. Each label's cut is the one that maximises F1 on the out-of-fold scores of the same setting, never on test; for a calibrated model it is about half the best F1 [6]. The test split is then scored once, with every Lecture 2 metric and the identity-group bias metrics of Borkan et al. (subgroup AUC, BPSN and BNSP) [2].`];
  if (!FIN) { out.push(pending("Final test results")); return out; }
  const T = FIN.test, RARE = ["identity_attack", "obscene", "threat", "sexual_explicit"];
  const pos = (t) => T[t].tp + T[t].fn;
  out.push(`Table ${tabNo + 1} gives the test results. The cuts lie between ${F(Math.min(...Object.values(FIN.cuts)), 2)} and ${F(Math.max(...Object.values(FIN.cuts)), 2)}, far below 0.5, because positives are rare. Toxicity and insult reach F1 ${F(T.toxicity.f1)} and ${F(T.insult.f1)}, with precision and recall both near one half; the four rare labels reach ${F(Math.min(...RARE.map((t) => T[t].f1)))} to ${F(Math.max(...RARE.map((t) => T[t].f1)))}, with wide intervals because the test split holds only ${N(Math.min(...RARE.map(pos)))} to ${N(Math.max(...RARE.map(pos)))} positives of each. ROC-AUC is between ${F(Math.min(...TARGETS.map((t) => T[t].roc_auc)))} and ${F(Math.max(...TARGETS.map((t) => T[t].roc_auc)))} for every label. The test F1 of every label is within ${F(Math.max(...TARGETS.map((t) => Math.abs(T[t].f1 - FIN.oof_best_f1[t]))), 2)} of its out-of-fold F1, so the cross-validation estimates carried over to unseen comments.`);
  return out;
}

function FINAL_TEXT2() {
  if (!FIN || !CMP) return [];
  const v = CMP.vs_phase3, m = v.mcnemar, out = [];
  out.push(`**Against Phase 3.** The Phase 3 MLP saved its toxicity scores on the same test comments, so the two models can be compared directly. On toxicity, PR-AUC rises from ${F(v.phase3.pr_auc)} to ${F(v.phase4.pr_auc)} (difference ${F(v.pr_auc_diff)}, 95% paired bootstrap interval ${F(v.pr_auc_diff_ci[0])} to ${F(v.pr_auc_diff_ci[1])}) and ROC-AUC from ${F(v.phase3.roc_auc)} to ${F(v.phase4.roc_auc)}. At each model's own max-F1 cut, F1 rises from ${F(v.phase3.f1)} to ${F(v.phase4.f1)}. McNemar's test on the two sets of predictions [3] finds ${N(m.only_phase4_right)} comments that only the new model gets right against ${N(m.only_phase3_right)} that only the old one does (χ² = ${F(m.chi2, 1)}, p < 0.001). The gain comes from three changes made in this phase: all training rows in a network designed for them, both text vectors at once, and one network per label with its own early stopping.`);
  const B = FIN.test_bias, P3S = P3.subgroups.identities;
  const groups = Object.keys(B).filter((g) => P3S[g] && P3S[g].mlp);
  const up = (k) => groups.filter((g) => B[g][k] > P3S[g].mlp[k]).length;
  const worst = Object.entries(B).sort((a, b) => a[1].bpsn_auc - b[1].bpsn_auc);
  out.push(`**Bias.** On the test split, for the ${groups.length} identity groups mentioned in at least ${N(100)} comments, BPSN AUC is higher than the Phase 3 MLP's for ${up("bpsn_auc")} groups, subgroup AUC for ${up("subgroup_auc")} and BNSP AUC for ${up("bnsp_auc")} [2]. The weakest groups are still "${worst[0][0]}" (BPSN ${F(worst[0][1].bpsn_auc)}, up from ${F(P3S[worst[0][0]].mlp.bpsn_auc)}) and "${worst[1][0]}" (${F(worst[1][1].bpsn_auc)}): non-toxic comments that mention them are still often scored above toxic comments that do not. The better ranking lifted almost every group, but it did not close the gap, and Section 4.6 showed that identity weights do not close it either.`);
  return out;
}

function CONCLUSION() {
  if (!FIN || !CMP) return pending("Conclusion");
  const v = CMP.vs_phase3;
  const worst = Object.entries(FIN.test_bias).sort((a, b) => a[1].bpsn_auc - b[1].bpsn_auc);
  return `We designed the final model by answering the Phase 3 feedback with evidence. Five-fold cross-validation on the merged pool, grouped by article, replaced the validation split, and every choice was made on out-of-fold scores. Focal loss was tested fairly, over ${s1Focal.length} combinations of γ and α, and matched BCE without beating it; the activation, the hidden layers, dropout, L2, the learning rate and the batch size were varied, and the base ReLU setting stayed best. The two changes that helped were giving the network both text vectors and training one network per label. The final model, six networks averaged over five seeds, beats the Phase 3 MLP on the same test comments (toxicity PR-AUC ${F(v.phase4.pr_auc)} against ${F(v.phase3.pr_auc)}, F1 ${F(v.phase4.f1)} against ${F(v.phase3.f1)}). Two limits remain: the rare labels reach F1 of only ${F(Math.min(...["identity_attack", "obscene", "threat", "sexual_explicit"].map((t) => FIN.test[t].f1)), 2)} to ${F(Math.max(...["identity_attack", "obscene", "threat", "sexual_explicit"].map((t) => FIN.test[t].f1)), 2)}, and non-toxic comments that mention some identities, above all "${worst[0][0]}" and "${worst[1][0]}", are still scored too high. The model therefore suits its intended use, ordering a moderator's queue, rather than removing comments on its own. The next phase builds the application of Section 5 around this model.`;
}
