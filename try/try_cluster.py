import os

text_to_write = "This is some text to write to the file"
file_path = "try2/try_sub/f.txt"

# Create or overwrite the file and write text to it
os.system('echo "{}" > {}'.format(text_to_write, file_path))
