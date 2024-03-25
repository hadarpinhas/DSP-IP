import cv2
import os

def divide_image_into_tiles(image_path, tile_width, tile_height):
    # Read the image
    img = cv2.imread(image_path)
    
    # Get the dimensions of the image
    img_height, img_width, _ = img.shape
    
    # Calculate the number of tiles in each dimension
    num_tiles_x = img_width // tile_width
    num_tiles_y = img_height // tile_height
    
    tiles = []
    for y in range(0, num_tiles_y):
        for x in range(0, num_tiles_x):
            # Compute the coordinates of the current tile
            x_start = x * tile_width
            y_start = y * tile_height
            x_end = x_start + tile_width
            y_end = y_start + tile_height
            
            # Extract the tile from the image
            tile = img[y_start:y_end, x_start:x_end]
            tiles.append(tile)
            
            # Optional: Save or display the tile
            # cv2.imshow(f'Tile {y*num_tiles_x + x}', tile)
            # cv2.waitKey(0)
            # cv2.imwrite(f'tile_{y}_{x}.jpg', tile)
    
    return tiles

# Example usage
# image_path = 'path/to/your/image.jpg'
# tile_width = 100  # width of each tile in pixels
# tile_height = 100  # height of each tile in pixels
# tiles = divide_image_into_tiles(image_path, tile_width, tile_height)

# # Optional: Display the number of tiles
# print(f"Total tiles: {len(tiles)}")



def getSorted(datasetDir, dirName, ext):
    """
    Filters and sorts files in a directory based on their numeric prefix and extension.
    Reads the sorted files differently based on the extension.

    Parameters:
    - datasetDir: The base directory where datasets are stored.
    - dir: The specific directory within datasetDir containing the files.
    - ext: The file extension to filter by.

    Returns:
    A list of data loaded from the sorted and filtered files.
    """
    unsortedList = os.listdir(os.path.join(datasetDir, dirName))

    # Filter out non-numeric filenames and ensure the file has the specified extension
    filtered_list = [x for x in unsortedList if x.split('.')[0].isdigit() and x.endswith(ext)]

    # Sort the filtered list
    sortedList = sorted(filtered_list, key=lambda x: int(x.split('.')[0]))

    dataList = []
    for file in sortedList:
        fullPath = os.path.join(datasetDir, dirName, file)
        if ext == '.jpg':
            # Read image file
            data = cv2.imread(fullPath)
        else:
            # Read non-image file as text
            with open(fullPath, 'r') as f:
                data = f.read()
        
        dataList.append(data)

    return dataList