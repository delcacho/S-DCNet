import requests
from bs4 import BeautifulSoup

html = requests.get("https://beachcam.meo.pt/livecams/").text
soup = BeautifulSoup(html)
seenCams = set()
for link in soup.find_all('a'):
    url = link.get('href') or ""
    if "livecams/" in url:
       seenCams.add(url)

print("url,playlist")
for cam in seenCams:
    url = "https://beachcam.meo.pt"+cam
    html = requests.get(url).text
    pos = html.find("var name = '")
    if pos < 0:
      continue
    pos += len("var name = '")
    pos2 = html.find("'",pos)
    print(url,"https://video-auth1.iol.pt/beachcam/"+html[pos:pos2]+"/playlist.m3u8",sep=",")
    
