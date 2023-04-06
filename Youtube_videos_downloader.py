from pytube import YouTube
from sys import argv

link = argv[1]
yt = YouTube(link)
yt.age_restricted


print("Autor ", yt.author)
print("Title ", yt.title)
print("Views ", yt.views)
print("Descrição ", yt.description)

yDownload = yt.streams.get_highest_resolution() 
pc_path=input("Where to send the video?\n")
yDownload.download(f"{pc_path}")

