from TFGeneration.GenerateTFRecord import *
import argparse
from TableGeneration.tools import html_to_img
from selenium.webdriver import Firefox
from selenium.webdriver.firefox.options import Options
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import uuid
from tqdm import tqdm
import os


# os.environ["PATH"] += os.pathsep + "/Users/aghiles/geckdriver" # path to geckdriver

def plot_table(table_pil_img, rows_coords, cols_coords, cells_coords):
    
    fig, ax = plt.subplots()

    plt.imshow(table_pil_img)

    for row_bbox in rows_coords:
        x = row_bbox[0]
        y = row_bbox[1]
        width = row_bbox[2] - row_bbox[0]
        height = row_bbox[3] - row_bbox[1]
        patch = Rectangle(xy=(x,y),width=width, height=height,
                        alpha = 0.5, edgecolor='red', facecolor='none')

        ax.add_patch(patch)

    for col_bbox in cols_coords:
        x = col_bbox[0]
        y = col_bbox[1]
        width = col_bbox[2] - col_bbox[0]
        height = col_bbox[3] - col_bbox[1]
        patch = Rectangle(xy=(x,y),width=width, height=height,
                        alpha = 0.5, edgecolor='blue', facecolor='none')

        ax.add_patch(patch)

    for cell_bbox in cells_coords:
        # cell_bbox = cell_bbox[2:]
        x = cell_bbox[0]
        y = cell_bbox[1]
        width = cell_bbox[2] - cell_bbox[0]
        height = cell_bbox[3] - cell_bbox[1]
        patch = Rectangle(xy=(x,y),width=width, height=height,
                        alpha = 0.5, edgecolor='green', facecolor='none')

        ax.add_patch(patch)

    plt.show()

if __name__ == '__main__':
    # parser=argparse.ArgumentParser()
    # parser.add_argument('--n-images', type=int,default=1)                #number of images in a single tfrecord file
    # parser.add_argument('--output',default='output/')               


    # args=parser.parse_args()
    
    n_images = 2000

    output = f"dataset_{n_images}"
    # shutil.rmtree(output)
    
    distributionfile='unlv_distribution'
    
    ANNOTATION_ROW_ID = 1
    ANNOTATION_COL_ID = 2
    ANNOTATION_CELL_ID = 3

    TEST_RATE = 0.2
    MIN_MAX_ROWS = [2, 8]
    MIN_MAX_COLS = [2, 8]
    CATEGORIES = [0, 1, 2]

    os.mkdir(output)
    os.mkdir(os.path.join(output, 'train'))
    os.mkdir(os.path.join(output, 'train', 'images'))

    os.mkdir(os.path.join(output, 'test'))
    os.mkdir(os.path.join(output, 'test', 'images'))


    train_images = []
    annotations = []


    test_set_ids = random.sample(range(n_images), int(n_images * TEST_RATE))

    train_images = []
    train_annotations = []

    test_images = []
    test_annotations = []

    categories = [{
                        'name': 'row',
                        'id' : ANNOTATION_ROW_ID, 
                        'category_id': ANNOTATION_ROW_ID,
                        'color' : "#FF0000",
                        'supercategory' : 'row'
                },
                {
                        'name': 'column',
                        'id' : ANNOTATION_COL_ID, 
                        'category_id': ANNOTATION_COL_ID,
                        'color' : "#0000FF",
                        'supercategory' : 'column'
                },
                {
                        'name': 'cell',
                        'id' : ANNOTATION_CELL_ID, 
                        'category_id': ANNOTATION_CELL_ID,
                        'color' : "#00FF00",
                        'supercategory' : 'cell'
                }
    ]
    # [lentext,txt,xmin,ymin,xmax,ymax]

    image_id = 1
    while image_id < n_images:

        print(f"creating image number: {image_id}\r", end='')

        opts = Options()
        opts.set_headless()

        is_test = (image_id in test_set_ids)

        try: 
            driver = Firefox(options=opts)

            table = Table(no_of_rows=np.random.randint(low=MIN_MAX_ROWS[0], high=MIN_MAX_ROWS[1] + 1), 
                        no_of_cols=np.random.randint(low=MIN_MAX_COLS[0], high=MIN_MAX_COLS[1] + 1), 
                        images_path='images',ocr_path='unlv_xml_ocr',
                        gt_table_path='unlv_xml_gt',
                        assigned_category=np.random.choice(CATEGORIES), distributionfile=distributionfile)

            cells_matrix, cols_matrix, rows_matrix, cells_idcounter, html, tablecategory = table.create()
            
            im, cells_bboxes, rows_bboxes, cols_bboxes = html_to_img(driver, html_content=html, id_count=cells_idcounter, table=table)
            driver.quit()

            # TODO: add killing firefox subprocess for linux machines
        except Exception as e:
            continue

        current_annotations = []
        
        image_id += 1
        
        # create image json object
        img_name = f"IMG-{image_id}.png"
        tmp_image = {
                "file_name": img_name,
                "id" : image_id,
                "width" : im.size[0],
                "height" : im.size[1]
            }

        # append rows
        for row_bbox in rows_bboxes:
            tmp_annotation =   {
                    "image_id": image_id,
                    "id": uuid.uuid1().int >> 64,
                    "bbox": [row_bbox[0], row_bbox[1], row_bbox[2] - row_bbox[0], row_bbox[3] - row_bbox[1]],
                    "area": (row_bbox[3] - row_bbox[1]) * (row_bbox[2] - row_bbox[0]),
                    "iscrowd": 0,
                    "bbox_mode": 1,
                    "category_id": ANNOTATION_ROW_ID
                }
            current_annotations.append(tmp_annotation)  
        
        # append columns
        for col_bbox in cols_bboxes:
            tmp_annotation =   {
                    "image_id": image_id,
                    "id": uuid.uuid1().int >> 64,
                    "bbox": [col_bbox[0], col_bbox[1], col_bbox[2] - col_bbox[0], col_bbox[3] - col_bbox[1]],
                    "area": (col_bbox[3] - col_bbox[1]) * (col_bbox[2] - col_bbox[0]),
                    "iscrowd": 0,
                    "bbox_mode": 1,
                    "category_id": ANNOTATION_COL_ID
                }
            current_annotations.append(tmp_annotation)  

        # append CELLS
        for cell_bbox in cells_bboxes:
            # cell_bbox = cell_bbox[2:]
            tmp_annotation =   {
                    "image_id": image_id,
                    "id": uuid.uuid1().int >> 64,
                    "bbox": [cell_bbox[0], cell_bbox[1], cell_bbox[2] - cell_bbox[0], cell_bbox[3] - cell_bbox[1]],
                    "area": (cell_bbox[3] - cell_bbox[1]) * (cell_bbox[2] - cell_bbox[0]),
                    "iscrowd": 0,
                    "bbox_mode": 1,
                    "category_id": ANNOTATION_CELL_ID
                }
            current_annotations.append(tmp_annotation) 

        # update global coco json objects
        if is_test:
            test_annotations.extend(current_annotations)
            test_images.append(tmp_image)
            im.save(os.path.join(output, "test",  "images", img_name))
        else:
            train_annotations.extend(current_annotations)
            train_images.append(tmp_image)
            im.save(os.path.join(output, "train",  "images", img_name))
        
        # plot_table(table_pil_img=im, rows_coords=rows_bboxes, cols_coords=cols_bboxes, cells_coords=cells_bboxes)
        # print(test_annotations)
        # print(train_annotations)
        # plt.show()
        # exit(1)

    test_dataset = {
                "images": list(test_images),
                "annotations": list(test_annotations),
                "categories": list(categories)
            }  

    train_dataset = {
                "images": list(train_images),
                "annotations": list(train_annotations),
                "categories": list(categories)
            }   
    open(os.path.join(output, "train", "train_dataset.json"),
                                "w").write(json.dumps(train_dataset, indent=2))
    open(os.path.join(output, "test", "test_dataset.json"),
                                "w").write(json.dumps(test_dataset, indent=2))
    
    
