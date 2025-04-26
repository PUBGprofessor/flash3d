# from pytubefix import YouTube
# import requests
# cookies = {
#     "LOGIN_INFO": "AFmmF2swRQIhALMW6CiLZWn4Pai8rGAFV754b6u0SBT6-lnDNw_bHZ6FAiBas-NiA2qRArXl5apar2UtmTYWIpgM0BUpWuE8M-D2sQ:QUQ3MjNmem1YQWZOWXRrVGNWVzByazRJSnZldzVwMy1JU2dsbmlhSU5NMVladWs2VDl6ZWdIWEJfVklXZ2tscl9PWGRsUjFMR2RZWWRwQXpjaWdhUDB2ZXNxUjhTUGs0c1BnVUpRQXNBUm9ZVVhzT2Zrc1B4WnREcHJ5bkJVNEp4aUZLdnlSVk5TM3ZmUm1veWMxTERtVUljWExJVXFkNlZR",
#     "HSID": "AONpr79DLkAmjIuKn",
#     "SSID": "AAsCBZPPCU981LyAZ",
#     "APISID": "Qng9uqfDlJ9-7otO/A_LGlV04TUE3gKo09",
#     "SAPISID": "MP18pf3tr9kjKV69/AvPDEEerVBtcXLQQY",
#     "__Secure-1PAPISID": "MP18pf3tr9kjKV69/AvPDEEerVBtcXLQQY",
#     "__Secure-3PAPISID": "MP18pf3tr9kjKV69/AvPDEEerVBtcXLQQY",
#     "SID": "g.a000vAjiqQ3fbqLBFM0dMsC1_8HFblAV2ap9Ngf8r0IScFgjEpk9lLmja1bXikLqc75btvJvHwACgYKAbASARUSFQHGX2Mi1IUA_gO9EWPJWS2w8yjkVBoVAUF8yKr4ZFKgtkKnipsunck7RV6Q0076",
#     "__Secure-1PSID": "g.a000vAjiqQ3fbqLBFM0dMsC1_8HFblAV2ap9Ngf8r0IScFgjEpk97aB4r5g-ywrV4VSy4NIKnAACgYKAWsSARUSFQHGX2Mi9deG4N0W9-oWENqEaW793BoVAUF8yKqbumu2VWjyfV5MjcfXu_EV0076",
#     "__Secure-3PSID": "g.a000vAjiqQ3fbqLBFM0dMsC1_8HFblAV2ap9Ngf8r0IScFgjEpk9MfWOSwzPD-sA5qt3xoIq4AACgYKAe4SARUSFQHGX2MizOFWIg0FwBk6e_vmbHIj3RoVAUF8yKp8ChMSbXH3Ceh63gDZ5zhu0076",
#     "PREF": "tz=Asia.Shanghai&f4=4000000",
#     "VISITOR_INFO1_LIVE": "OLzMrBN66JY",
#     "YSC": "Y8LaW9KxTdc",
#     "__Secure-ROLLOUT_TOKEN": "COyD0b7S6dX6ehDnybvqle6KAxjNxYWOvuCMAw%3D%3D"
# }
# # url = "https://www.youtube.com/watch?v=21d49eTFWc4"  # Replace with your YouTube URL
# url = "https://www.youtube.com/watch?v=F3CmzAExWFU "  # Replace with your YouTube URL
# current_file = "tmpdir/F3CmzAExWFU"  # Replace with your desired file name
# # Create a session with cookies
# # session = requests.Session()
# # session.cookies.update(cookies)
# # response = session.get(url)
# yt = YouTube(url, use_oauth=True)
# # yt = YouTube(url, use_oauth=False, allow_oauth_cache=True, use_po_token=True)
# stream = yt.streams.filter(res='360p').first()
# stream.download(str(current_file))

import yt_dlp
import subprocess
import sys
# 设置视频 URL 和保存路径
url = "https://www.youtube.com/watch?v=W8tI5CnpczQ"  # YouTube 视频链接
current_file = "tmpdir/W8tI5CnpczQ.mp4"  # 保存文件的路径（注意加上文件扩展名）

# 配置 yt-dlp 下载选项
ydl_opts = {
    'format': 'bestvideo[height=360]',  # 下载 360p 视频
    'outtmpl': current_file,  # 指定输出文件路径
    'noplaylist': True,  # 不下载播放列表中的视频
    "cookies": r'test_code\www.youtube.com_cookies.txt',
    # 'progress_hooks': [lambda d: print(d)]  # 可选：打印下载进度
}
def download_video(url, cookies_path, output_path):
    # yt-dlp 命令
    ydl_cmd = [
        'yt-dlp', 
        '--cookies', cookies_path,  # cookies 文件路径
        '--format', 'bestvideo[height=360]',  # 下载 360p 视频
        '--output', output_path,  # 设置输出路径
        '--no-check-certificate',  # 不验证证书
        url  # YouTube 视频 URL
    ]
    
    try:
        # 调用 yt-dlp 命令
        result = subprocess.run(ydl_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e.stderr}", file=sys.stderr)
# 使用 yt-dlp 下载视频
# with yt_dlp.YoutubeDL(ydl_opts) as ydl:
#     ydl.download([url])

download_video(url, r'./test_code/www.youtube.com_cookies.txt', r"F:/3DGS_code/flash3d/tmpdir/1.mp4")
