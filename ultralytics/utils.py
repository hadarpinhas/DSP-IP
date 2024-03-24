import cv2

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
