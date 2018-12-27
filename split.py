from moviepy.editor import *

N = 10

text_file = open("result.txt", "r")
results = text_file.read().split('\n')

for i in range (len(results)-2):
    clip_start = int(results[i]) * N / float(25)
    clip_end = int(results[i+1]) * N / float(25)
    clip = VideoFileClip("test.mp4").subclip(clip_start, clip_end)
    clip.write_videofile("./output/shot_%s.mp4" %i)