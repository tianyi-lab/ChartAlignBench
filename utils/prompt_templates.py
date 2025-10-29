text_grounding_prompt_template = """Can you identify the text style for the chart image image_1_tag. Provide the answer as JSON in this format:-
{
  "chart title": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart legend": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart axes labels": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  },
  "chart axes ticks": {
    "size": <numerical value (pt)>,
    "weight": <"light" | "normal" | "bold">,
    "fontfamily": <"sans-serif" | "serif" | "cursive" | "fantasy" | "monospace">
  }
}
"""

attribute_alignment_prompt_template = "Given attr_change_gt_type_tag information for 2 chart images. Chart-1:-image_1_predicted_grounding_tag\nChart-2:-image_2_predicted_grounding_tag\nThe charts differ in attr_change_gt_type_tag. Can you identify the change? Mention the final answer strictly of form: attr_change_ans_format_tag, no explaination required."

attribute_answer_format_dict = {
    "color": "JSON {<attribute 1 json object>.....<attribute k json object> for all attributes i = 1 to k with COLOR CHANGE} where the <attribute i json object> format is: {\"<attribute name>\": {\"initial value\": <color from chart-1>, \"modified value\": <color from chart-2>}} Only include attributes where 'initial value' differs from 'modified value'",    
    "legend": "{\"position\": {\"initial value\": <legend position in chart-1>, \"modified value\": <legend position in chart-2>}}",
    "text_style": "{<chart-section-i>: <chart-section-i change json> for chart-section-i in [\"chart title\", \"chart legend\", \"chart axes\", \"chart ticks\"]} where <chart-section-i change json> is json form: {<text characteristic>: {\"initial value\": <value in chart-1>, \"modified value\": <value in chart-2>} for <text characteristic> in [\"size\", \"weight\", \"fontfamily\"]}." 
}

PROMPT_TEMPLATES = {
    "data": {
        "grounding": "Given a chart image: image_1_tag\nGenerate the table (csv format) for the chart data. Only output the table directly.",
        "alignment": "Given table (csv format) for first chart:-\nimage_1_predicted_grounding_tag\nGiven table (csv format) for second chart:-\nimage_2_predicted_grounding_tag\nThe second chart differs from first due to change in value of cells_change_cnt_tag of the cell(s). Can you identify the cells_change_cnt_tag cell(s)? Mention final answer of form: \"[<cell i json> for for all i cells with value change]\" where json is of form: {\"row name\": <row name for the cell>, \"column name\": <column name for the cell>, \"value in chart 1\": <cell value in chart 1>, \"value in chart 2\": <cell value in chart 2>}. Only cells_change_cnt_tag cell(s) hence output only cells_change_cnt_tag json in list, no explaination needed.",
    },
    "attribute": {
        "color": {
            "grounding": "Can you list the attributes with unique colors in the chart image:- image_1_tag? Mention final answer of form: list[for all attributes 'i' -> {\"attribute\": <attribute 'i' name>, \"color\": <attribute 'i' color <hex color value>>}]. Mention only final answer, no explanation required.",
        },
        "legend": {
            "grounding": "Can you identify the legend position for the chart image:- image_1_tag? Mention answer of form json: {\"legend position\": <position of legend in the chart>}\". Position of legend is not the order of items in legend but instead position of the legend box in chart. There are 9 possible values:- ['upper right', 'upper left', 'lower left', 'lower right', 'center left', 'center right', 'lower center', 'upper center', 'center'].",
        },
        "text_style": {
            "grounding": text_grounding_prompt_template,
        }
    }
}

PROMPT_TEMPLATES["attribute"]["color"]["alignment"] = attribute_alignment_prompt_template.replace("attr_change_gt_type_tag", "color").replace("attr_change_ans_format_tag", attribute_answer_format_dict["color"])
PROMPT_TEMPLATES["attribute"]["legend"]["alignment"] = attribute_alignment_prompt_template.replace("attr_change_gt_type_tag", "legend").replace("attr_change_ans_format_tag", attribute_answer_format_dict["legend"])
PROMPT_TEMPLATES["attribute"]["text_style"]["alignment"] = attribute_alignment_prompt_template.replace("attr_change_gt_type_tag", "text_style").replace("attr_change_ans_format_tag", attribute_answer_format_dict["text_style"])
