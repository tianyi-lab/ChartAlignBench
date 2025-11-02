import Levenshtein
import numpy as np
import re
from scipy.optimize import linear_sum_assignment

from utils.SCRM import csv_eval

# Grounding (csv metric, color: L2 error in RGB space, text style: direct accuracy, legend: correct/incorrect)
# Robustness

################################
# DATA GROUNDING (CSV Precision using SCRM metric)

def get_data_grounding_csv_precision(gt_csv_str, pred_csv_str):

    # remove brackets (like "million")
    gt_csv_str = re.sub(r'\(.*?\)', '', gt_csv_str).strip()
    pred_csv_str = re.sub(r'\(.*?\)', '', pred_csv_str).strip()

    pred_csv_str = pred_csv_str.replace("\t", "\\t").replace("\n", "\\n")
    pred_csv_str += "\\n"

    gt_csv_str = gt_csv_str.replace("\t", "\\t").replace("\n", "\\n")
    gt_csv_str += "\\n"

    response = csv_eval([gt_csv_str], [pred_csv_str], 0)

    # print(gt_csv_str)
    # print(pred_csv_str)
    # print(response)

    # MAP (mean average precision) for high tolerance
    return round(response[3], 3)

################################


################################
# COLOR GROUNDING (L2 distance in RGB space)

def hex_to_rgb(h):
    """Convert hex color (e.g. '#00aaff') to RGB array."""
    h = h.lstrip('#')
    return np.array([int(h[i:i+2], 16) for i in (0, 2, 4)])

def rgb_l2(c1, c2):
    """Compute L2 distance between two hex colors."""
    return np.linalg.norm(hex_to_rgb(c1) - hex_to_rgb(c2))

def levenshtein_sim(a, b):
    """Normalized Levenshtein similarity between two strings."""
    return 1 - Levenshtein.distance(a, b) / max(len(a), len(b))

def get_color_grounding_distance(gt, pred, threshold=0.5, penalty_value=441.67):
    """
    Compare ground truth and predicted color grounding using optimal Levenshtein-based mapping.
    
    Args:
        gt (dict): ground truth mapping {encoding_name: hex_color}
        pred (dict): predicted mapping {encoding_name: hex_color}
        threshold (float): minimum similarity to consider a valid match (default 0.5)
        penalty_value (float): penalty for unmatched keys (max RGB L2 distance ≈ 441.67)
    
    Returns:
        dict with:
            - avg_matched_l2: average distance over matched pairs only
            - avg_with_penalty: includes penalty for unmatched pairs
            - matched_pairs: {gt_key: pred_key}
            - distances: {gt_key: rgb_l2_distance}
    """

    try: 
        gt_keys = list(gt.keys())
        pred_keys = list(pred.keys())

        # Build cost matrix (negative similarity → cost)
        sim_matrix = np.zeros((len(gt_keys), len(pred_keys)))
        for i, gk in enumerate(gt_keys):
            for j, pk in enumerate(pred_keys):
                sim_matrix[i, j] = levenshtein_sim(gk, pk)
        
        # Convert similarity to cost for Hungarian algorithm (maximize sim = minimize -sim)
        cost_matrix = -sim_matrix

        # Solve optimal assignment
        gt_idx, pred_idx = linear_sum_assignment(cost_matrix)

        matched_pairs = {}
        distances = {}

        # Keep only matches above threshold
        for i, j in zip(gt_idx, pred_idx):
            sim = sim_matrix[i, j]
            if sim >= threshold:
                gk, pk = gt_keys[i], pred_keys[j]
                d = rgb_l2(gt[gk], pred[pk])
                matched_pairs[gk] = pk
                distances[gk] = d

        # Compute averages
        if distances:
            avg_matched = np.mean(list(distances.values()))
        else:
            avg_matched = None

        # Penalty for unmatched gt keys
        unmatched_count = len(gt_keys) - len(distances)
        if distances:
            avg_with_penalty = (np.sum(list(distances.values())) + unmatched_count * penalty_value) / len(gt_keys)
        else:
            avg_with_penalty = penalty_value  # all unmatched

        return {
            "avg_matched_l2": avg_matched,
            "avg_with_penalty": avg_with_penalty,
            "matched_pairs": matched_pairs,
            "distances": distances,
        }

    except: 
        # ERROR: Incorrect JSON format
        return {
            "avg_matched_l2": "Incorrect JSON formatting",
            "avg_with_penalty": "Incorrect JSON formatting",
            "matched_pairs": "Incorrect JSON formatting",
            "distances": "Incorrect JSON formatting",
        }

################################

################################
# TEXT STYLE GROUNDING (Accuracy for size/weight/text style)

def get_text_style_grounding_accuracy(gt, pred, size_margin=0.1):
    """
    Calculate attribute-wise accuracy for text style grounding.
    
    Args:
        gt (dict): Ground truth JSON mapping region -> {size, weight, fontfamily}.
        pred (dict): Model prediction JSON in the same format.
        size_margin (float): Allowed percentage margin for size (default ±10%).
    
    Returns:
        dict: Accuracy per attribute (size, weight, fontfamily) and overall.
    """

    try:
        attr_names = ["size", "weight", "fontfamily"]
        correct = {a: 0 for a in attr_names}
        total = {a: 0 for a in attr_names}

        for region, gt_attrs in gt.items():
            if region not in pred:
                # skip missing region in prediction
                continue
            pred_attrs = pred[region]
            for attr in attr_names:
                if attr not in gt_attrs or attr not in pred_attrs:
                    continue
                g, p = gt_attrs[attr], pred_attrs[attr]

                if attr == "size":
                    # numeric comparison with margin
                    if g > 0 and abs(g - p) <= size_margin * g:
                        correct[attr] += 1
                else:
                    # categorical comparison
                    if g == p:
                        correct[attr] += 1
                total[attr] += 1

        # compute accuracies
        acc = {a: round((correct[a] / total[a]), 3) if total[a] > 0 else None for a in attr_names}

        # overall (mean over available attributes)
        valid_accs = [v for v in acc.values() if v is not None]
        acc["overall"] = round((sum(valid_accs) / len(valid_accs)), 3) if valid_accs else None

        return acc

    except: 
        # ERROR: Incorrect JSON format
        return "Incorrect JSON formatting"

################################


################################
# GROUNDING METRICS 

def compute_grounding_metrics(ground_truth, pred, content_type):

    if content_type == "data":
        return get_data_grounding_csv_precision(ground_truth, pred)

    elif content_type == "color":
        return get_color_grounding_distance(ground_truth, pred)["avg_matched_l2"]

    elif content_type == "text_style":
        return get_text_style_grounding_accuracy(ground_truth, pred)

    else:
        return "Incorrect content type"
################################


################################
# DATA ALIGNMENT CALCULATION
# -----------------------------
# Helper Functions
# -----------------------------
def remove_units_in_brackets(s):
    if not isinstance(s, str):
        return str(s)
    import re
    return re.sub(r'\(.*?\)', '', s).strip()

def get_float_val(val):
    try:
        return float(str(val).replace(',', '').strip())
    except:
        return float('inf')

def compute_name_similarity(gt_json, gr_json):
    row_sim = Levenshtein.ratio(remove_units_in_brackets(gt_json["row name"]).lower(),
                                remove_units_in_brackets(gr_json["row name"]).lower())
    col_sim = Levenshtein.ratio(remove_units_in_brackets(gt_json["column name"]).lower(),
                                remove_units_in_brackets(gr_json["column name"]).lower())
    return (row_sim + col_sim) / 2.0  # simple average


def compute_value_precision(gt_json, gr_json):
    """Compute how close the numeric values are (0–1 scale)."""
    # Handle missing values
    for key in ["value in chart 1", "value in chart 2"]:
        if gr_json.get(key) is None:
            gr_json[key] = float('inf')

    # Convert to float
    for key in ["value in chart 1", "value in chart 2"]:
        if isinstance(gt_json[key], str):
            gt_json[key] = get_float_val(gt_json[key])
        if isinstance(gr_json[key], str):
            gr_json[key] = get_float_val(gr_json[key])

    try:
        diff1 = abs(gr_json["value in chart 1"] - gt_json["value in chart 1"]) / (abs(gt_json["value in chart 1"]) + 1e-8)
    except:
        diff1 = 1.0
    try:
        diff2 = abs(gr_json["value in chart 2"] - gt_json["value in chart 2"]) / (abs(gt_json["value in chart 2"]) + 1e-8)
    except:
        diff2 = 1.0

    avg_diff = (diff1 + diff2) / 2.0
    return max(0.0, 1.0 - avg_diff)  # higher is better


def compute_similarity_matrix(gt_list, gr_list):
    """Compute similarity matrix (GT × Generated) for detection (row+col)."""
    sim_matrix = np.zeros((len(gt_list), len(gr_list)))
    for i, gt in enumerate(gt_list):
        for j, gr in enumerate(gr_list):
            sim_matrix[i, j] = compute_name_similarity(gt, gr)
    return sim_matrix


def get_best_matching_pairs(sim_matrix):
    """Use Hungarian algorithm to find best one-to-one matching."""
    # Convert to cost matrix (since the algorithm minimizes)
    cost_matrix = 1.0 - sim_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    return list(zip(row_ind, col_ind))  # (gt_idx, gr_idx)


# -----------------------------
# Core Scoring Functions
# -----------------------------
def compute_detection_score(sim_matrix, threshold=0.8):
    """Compute F1 score for detection (row+col name similarity)."""
    m, n = sim_matrix.shape
    pairs = get_best_matching_pairs(sim_matrix)

    TP = sum(sim_matrix[i, j] >= threshold for i, j in pairs)
    FP = n - TP
    FN = m - TP

    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = (2 * precision * recall) / (precision + recall + 1e-8)

    return f1, TP, FP, FN, pairs


def compute_precision_score(gt_list, gr_list, pairs, sim_matrix, threshold=0.8):
    """Compute average value precision for correctly detected pairs."""
    precisions = []
    for i, j in pairs:
        if sim_matrix[i, j] >= threshold:
            precisions.append(compute_value_precision(gt_list[i], gr_list[j]))

    if len(precisions) == 0:
        return 0.0
    return np.mean(precisions)


def get_data_alignment_score(gt_json_dict, gr_json_dict, threshold=0.5, alpha=0.5):

    print(gt_json_dict)
    print(gr_json_dict)

    """Compute detection, precision, and combined alignment scores."""
    gt_list = [gt_json_dict[k] for k in gt_json_dict.keys()]
    gr_list = [gr_json_dict[k] for k in gr_json_dict.keys()]

    sim_matrix = compute_similarity_matrix(gt_list, gr_list)
    detection_f1, TP, FP, FN, pairs = compute_detection_score(sim_matrix, threshold)
    precision_score = compute_precision_score(gt_list, gr_list, pairs, sim_matrix, threshold)

    total_score = 10 * (alpha * detection_f1 + (1 - alpha) * precision_score)

    return {
        "detection_f1": detection_f1,
        "precision_score": precision_score,
        "total_score": round(total_score, 1),
        "TP": TP,
        "FP": FP,
        "FN": FN
    }

################################




################################
# COLOR ALIGNMENT CALCULATION

def hex_to_rgb(hex_color):
    """Convert hex color (e.g., '#ff0000') to RGB tuple (0-255)."""
    hex_color = hex_color.strip().lstrip('#')
    if len(hex_color) == 3:  # short form like #f00
        hex_color = ''.join([c * 2 for c in hex_color])
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def color_distance(c1, c2):
    """Compute Euclidean distance in RGB space."""
    rgb1, rgb2 = np.array(c1), np.array(c2)
    return np.linalg.norm(rgb1 - rgb2)

def match_encodings(gt_keys, pred_keys, threshold=0.5):
    """
    Match predicted encodings to GT encodings using Levenshtein ratio.
    Returns mapping: pred_key -> best_gt_key (if above threshold).
    """
    matched = {}
    used_gt = set()
    for pk in pred_keys:
        best_match = None
        best_score = 0
        for gk in gt_keys:
            if gk in used_gt:
                continue
            score = Levenshtein.ratio(pk.lower(), gk.lower())
            if score > best_score:
                best_match, best_score = gk, score
        if best_score >= threshold:
            matched[pk] = best_match
            used_gt.add(best_match)
    return matched

def get_color_alignment_score(gt_json, pred_json, alpha=0.5, threshold=0.5):
    """
    Compute color alignment score between GT and predicted JSON.
    Inputs:
        gt_json: dict {encoding: {"initial value": "#xxxxxx", "modified value": "#xxxxxx"}}
        pred_json: dict of same form
    Returns:
        dict with precision, recall, f1, color_accuracy, final_score
    """
    try:

        # --- Step 1: Preprocess keys ---
        # in ground truth: if encodings with no value change, remove them
        gt_dict = {remove_units_in_brackets(k): v for k, v in gt_json.items() if v["initial value"] != v["modified value"]}
        pred_dict = {remove_units_in_brackets(k): v for k, v in pred_json.items()}

        gt_keys = list(gt_dict.keys())
        pred_keys = list(pred_dict.keys())

        # --- Step 2: Match predicted encodings to GT ---
        matched = match_encodings(gt_keys, pred_keys, threshold)

        # --- Step 3: Classify encodings ---
        tp, fp, fn = [], [], []

        matched_gt = set(matched.values())
        for pk in pred_keys:
            if pk in matched:
                tp.append((matched[pk], pk))
            else:
                fp.append(pk)
        for gk in gt_keys:
            if gk not in matched_gt:
                fn.append(gk)

        # --- Step 4: Compute detection metrics ---
        precision = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) > 0 else 0.0
        recall = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        # --- Step 5: Compute color accuracy for TP ---
        color_accs = []
        dmax = np.sqrt(3 * 255**2)
        for gk, pk in tp:
            gt_colors = gt_dict[gk]
            pred_colors = pred_dict[pk]
            try:
                d_init = color_distance(hex_to_rgb(gt_colors["initial value"]), hex_to_rgb(pred_colors["initial value"]))
                d_mod = color_distance(hex_to_rgb(gt_colors["modified value"]), hex_to_rgb(pred_colors["modified value"]))
                color_acc = 1 - ((d_init + d_mod) / (2 * dmax))
                color_accs.append(max(color_acc, 0))
            except Exception:
                continue
        color_accuracy = np.mean(color_accs) if color_accs else 0.0

        # --- Step 6: Final score (normalized 0–10) ---
        final_score = 10 * (alpha * f1 + (1 - alpha) * color_accuracy)

        return {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "color_accuracy": round(color_accuracy, 4),
            "total_score": round(final_score, 1),
            "num_tp": len(tp),
            "num_fp": len(fp),
            "num_fn": len(fn)
        }

    except: 
        # ERROR: Incorrect JSON format
        return {
            "total_score": "Incorrect JSON formatting",
        }


####################################

################################
# TEXT ALIGNMENT CALCULATION

# Predefined allowed values
size_min, size_max = 8, 22
weight_options_list = ["light", "normal", "bold"]
fontfamily_options_list = ['sans-serif', 'serif', 'cursive', 'fantasy', 'monospace']

def normalize_label(label):
    """Normalize region/characteristic labels (case, whitespace)."""
    return label.strip().lower()

def get_text_style_alignment_score(gt_json, pred_json, alpha=0.5):
    """
    Evaluate text-style alignment between GT and predicted JSONs.

    Each JSON is of form:
    {
      "<text region>": {
          "<characteristic>": {"initial value": ..., "modified value": ...},
          ...
      },
      ...
    }

    Returns dict with: precision, recall, f1, char_accuracy, final_score.
    """

    # --- Step 1: Flatten into (region, characteristic) pairs ---
    def flatten_style_dict(d):
        flat = {}
        for region, chars in d.items():
            for ch_name, vals in chars.items():
                key = (normalize_label(region), normalize_label(ch_name))
                flat[key] = vals
        return flat

    gt_flat_in = flatten_style_dict(gt_json)

    gt_flat = {}
    for index, (key, value) in enumerate(gt_flat_in.items()):
        if value["initial value"] != value["modified value"]:
            gt_flat[key] = value

    pred_flat = flatten_style_dict(pred_json)

    gt_keys = list(gt_flat.keys())
    pred_keys = list(pred_flat.keys())

    # --- Step 2: Detection classification ---
    gt_set = set(gt_keys)
    pred_set = set(pred_keys)

    tp_keys = gt_set & pred_set
    fp_keys = pred_set - gt_set
    fn_keys = gt_set - pred_set

    precision = len(tp_keys) / (len(tp_keys) + len(fp_keys)) if (len(tp_keys) + len(fp_keys)) > 0 else 0.0
    recall = len(tp_keys) / (len(tp_keys) + len(fn_keys)) if (len(tp_keys) + len(fn_keys)) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # --- Step 3: Compute per-TP characteristic correctness ---
    def score_characteristic(ch_name, gt_vals, pred_vals):
        """Return value similarity score ∈ [0,1] for given characteristic."""
        if ch_name == "size":
            # Use absolute difference normalized to [0,1]
            g_i, g_m = gt_vals["initial value"], gt_vals["modified value"]
            p_i, p_m = pred_vals["initial value"], pred_vals["modified value"]
            diff = min((1.0 * abs(g_i - p_i) / g_i), 1.0) + min((1.0 * abs(g_m - p_m) / g_m), 1.0)
            return ((2 - diff) / 2)

        elif ch_name == "weight":
            # Map weight to ordinal
            def map_weight(w):
                if w not in weight_options_list:
                    return weight_options_list.index("normal")
                return weight_options_list.index(w)
            g_i, g_m = map_weight(gt_vals["initial value"]), map_weight(gt_vals["modified value"])
            p_i, p_m = map_weight(pred_vals["initial value"]), map_weight(pred_vals["modified value"])
            return ((g_i == p_i) + (g_m == p_m)) / 2

        elif ch_name == "fontfamily":
            g_i, g_m = gt_vals["initial value"].lower(), gt_vals["modified value"].lower()
            p_i, p_m = pred_vals["initial value"].lower(), pred_vals["modified value"].lower()
            return ((g_i == p_i) + (g_m == p_m)) / 2

        else:
            return 0.0

    char_accs = []
    for key in tp_keys:
        region, ch_name = key
        try:
            s_val = score_characteristic(ch_name, gt_flat[key], pred_flat[key])
            char_accs.append(s_val)
        except Exception:
            continue

    char_accuracy = np.mean(char_accs) if char_accs else 0.0

    # --- Step 4: Final score ---
    final_score = 10 * (alpha * f1 + (1 - alpha) * char_accuracy)

    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "char_accuracy": round(char_accuracy, 4),
        "total_score": round(final_score, 1),
        "num_tp": len(tp_keys),
        "num_fp": len(fp_keys),
        "num_fn": len(fn_keys)
    }

################################


################################
# LEGEND ALIGNMENT CALCULATION

# Define positions on a 3x3 grid
position_grid = {
    'upper left': (0, 0),
    'upper center': (1, 0),
    'upper right': (2, 0),
    'center left': (0, 1),
    'center': (1, 1),
    'center right': (2, 1),
    'lower left': (0, 2),
    'lower center': (1, 2),
    'lower right': (2, 2),
}

# Function to compute Manhattan distance between two positions
def manhattan_distance(pos1, pos2):
    coord1 = position_grid.get(pos1)
    coord2 = position_grid.get(pos2)
    if coord1 is None or coord2 is None:
        return None  # Handle invalid labels
    return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1])

# Function to convert distance to score
def score_from_distance(distance):
    if distance is None or distance > 4:
        return 0
    return max(5 - distance, 0)
        # making distance penalty higher -> better statistical significance (else all scores in range 8 to 10)

def get_legend_alignment_score(ground_truth_json_dict, predicted_res_json_dict):

    score = 0.

    initial_gt = ground_truth_json_dict['position']['initial value']
    initial_gen = predicted_res_json_dict['position']['initial value']
    modified_gt = ground_truth_json_dict['position']['modified value']
    modified_gen = predicted_res_json_dict['position']['modified value']

    score += score_from_distance(manhattan_distance(initial_gt, initial_gen))
    score += score_from_distance(manhattan_distance(modified_gt, modified_gen))

    return score

################################


################################
# ALIGNMENT SCORING FUNCTION (Calculate score)

def compute_alignment_score(ground_truth_json_dict, predicted_res_json_dict, content_type):

    if content_type == "data":
        return get_data_alignment_score(ground_truth_json_dict, predicted_res_json_dict)["total_score"]

    elif content_type == "color":
        return get_color_alignment_score(ground_truth_json_dict, predicted_res_json_dict)["total_score"]

    elif content_type == "text_style":
        return get_text_style_alignment_score(ground_truth_json_dict, predicted_res_json_dict)["total_score"]

    elif content_type == "legend":
        return get_legend_alignment_score(ground_truth_json_dict, predicted_res_json_dict)

    else:
        return "Incorrect content type"
################################
