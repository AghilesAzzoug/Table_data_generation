import traceback


from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import Firefox
from selenium.webdriver import PhantomJS

from selenium.webdriver.firefox.options import Options
from PIL import Image
from io import BytesIO
import warnings
import json
import time
def warn(*args, **kwargs):
    pass

warnings.warn = warn



def get_coords_for_rows(cells_in_rows, cells_bboxes):

    bboxes = []

    very_min_x = 999999
    very_max_x = 0

    for current_cells_in_rows in cells_in_rows:
        x_min = 999999
        y_min = 999999
        x_max = 0
        y_max = 0

        # [lentext,txt,xmin,ymin,xmax,ymax]
        for cell_id, cell_bbox in enumerate(cells_bboxes):
            if cell_id in current_cells_in_rows:
                x_min = min(x_min, cell_bbox[2])
                y_min = min(y_min, cell_bbox[3])

                x_max = max(x_max, cell_bbox[4])
                y_max = max(y_max, cell_bbox[5])

                very_min_x = min(x_min, very_min_x)
                very_max_x = max(x_max, very_max_x)

        bboxes.append([x_min, y_min, x_max, y_max])
    
    bboxes_ = []

    for box in bboxes:
        box[0] = very_min_x
        box[2] = very_max_x

        bboxes_.append(box)

    return bboxes_


def get_coords_for_cols(cells_in_cols, cells_bboxes):
    bboxes = []

    very_min_y = 999999
    very_max_y = 0

    for current_cells_in_cols in cells_in_cols:

        x_min = 999999
        y_min = 999999
        x_max = 0
        y_max = 0

        # [lentext,txt,xmin,ymin,xmax,ymax]
        for cell_id, cell_bbox in enumerate(cells_bboxes):
            
            if cell_id in current_cells_in_cols:
                x_min = min(x_min, cell_bbox[2])
                y_min = min(y_min, cell_bbox[3])

                x_max = max(x_max, cell_bbox[4])
                y_max = max(y_max, cell_bbox[5])

                # print(x_min, x_max, y_min, y_max)

                very_min_y = min(y_min, very_min_y)
                very_max_y = max(y_max, very_max_y)

        bboxes.append([x_min, y_min, x_max, y_max])
    
    bboxes_ = []

    for box in bboxes:
        box[1] = very_min_y
        box[3] = very_max_y

        bboxes_.append(box)

    return bboxes_


def get_coords_for_cells(words_in_cells, cells_bboxes):
    bboxes = []


    for current_words_in_cells in words_in_cells:

        x_min = 999999
        y_min = 999999
        x_max = 0
        y_max = 0

        # [lentext,txt,xmin,ymin,xmax,ymax]
        for cell_id, cell_bbox in enumerate(cells_bboxes):
            
            if cell_id in current_words_in_cells:
                x_min = min(x_min, cell_bbox[2])
                y_min = min(y_min, cell_bbox[3])

                x_max = max(x_max, cell_bbox[4])
                y_max = max(y_max, cell_bbox[5])

        bboxes.append([x_min, y_min, x_max, y_max])

    return bboxes

def html_to_img(driver, html_content, id_count, table):
    
    '''converts html to image'''
    counter=1                #This counter is to keep track of the exceptions and stop execution after 10 exceptions have occurred
    while(True):
        try:
            driver.get("data:text/html;charset=utf-8," + html_content)
            window_size=driver.get_window_size()
            max_height,max_width=window_size['height'],window_size['width']
            element = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, '0')))

            cells_bboxes = []
            # rows_bboxes = []
            # cols_bboxes = []
            max_x_window = 0
            max_y_window = 0
            for id in range(id_count):
                #e = driver.find_element_by_id(str(id))
                e = WebDriverWait(driver, 15).until(EC.presence_of_element_located((By.ID, str(id))))
                txt=e.text.strip()
                lentext=len(txt)
                loc = e.location
                size_ = e.size
                xmin = loc['x']
                ymin = loc['y']
                xmax = int(size_['width'] + xmin)
                ymax = int(size_['height'] + ymin)
                cells_bboxes.append([lentext,txt,xmin,ymin,xmax,ymax])

                max_x_window = max(max_x_window, xmax)
                max_y_window = max(max_y_window, ymax)
                # cv2.rectangle(im,(xmin,ymin),(xmax,ymax),(0,0,255),2)

            png = driver.get_screenshot_as_png()

            im = Image.open(BytesIO(png))

            im = im.crop((0, 0, max_x_window + 10, max_y_window + 10))

            rows_bboxes = get_coords_for_rows(cells_in_rows=table.all_rows, cells_bboxes=cells_bboxes)
            cols_bboxes = get_coords_for_cols(cells_in_cols=table.all_cols, cells_bboxes=cells_bboxes)
            cells_bboxes = get_coords_for_cells(words_in_cells=table.all_cells, cells_bboxes=cells_bboxes)

            return im, cells_bboxes, rows_bboxes, cols_bboxes

        except Exception as e:
            traceback.print_exc()

            counter+=1
            if(counter==10):
                raise e
            continue
            # return None