import json

def process_pred_csv(csv_str):
    # "csv" in beginning for some models
    if csv_str[0] == '\n':
         csv_str = csv_str[1:]

    csv_str = csv_str.replace('csv\n', '').replace('-', '').replace('|','\t')

    # replace commas with \t (standard) but exclude commas in numerical values
    csv_str = csv_str.replace(',000', '#000').replace(',', '\t').replace('#000', '000')

    return csv_str


def get_chart_pair_images(img):

    width, height = img.size
    midpoint = width // 2

    left_box = (0, 0, midpoint, height)
    chart_1_img = img.crop(left_box)

    right_box = (midpoint, 0, width, height)
    chart_2_img = img.crop(right_box)

    return chart_1_img, chart_2_img


def process_attr_alignment_response(response, attribute_altered):

    # if color, extra square brackets
    return response.replace("```json","").replace("```","").replace('[', '').replace(']', '')

def data_alignment_str_to_json(predicted_alignment, num_cell_difference):

    predicted_alignment = predicted_alignment.replace('\n', '')
    predicted_alignment = predicted_alignment[(predicted_alignment.find('[')) : (predicted_alignment.rfind(']') + 1)]
    
    predicted_alignment_json = None
    try:
        # convert string to Python list
        predicted_alignment_list = json.loads(predicted_alignment)

        # convert list to indexed dictionary
        predicted_alignment_json = {i: item for i, item in enumerate(predicted_alignment_list)}
    except:
        dummy_json_element = {"row name": "sample row", "column name": "sample column", "value in chart 1": float('inf'), "value in chart 2": float('inf')}
        predicted_alignment_json = {i: dummy_json_element for i in range(num_cell_difference)}

    return predicted_alignment_json

