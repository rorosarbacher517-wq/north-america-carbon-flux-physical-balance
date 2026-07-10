# #  modis_braf_reflectance
import os

# 建议挂代理，如Clash US节点，速度更快
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"

from datetime import datetime, timedelta
from multiprocessing import Pool

import pandas as pd
import requests
from bs4 import BeautifulSoup
import os

# tiles = "h13v01".split(",")
tiles = "h02v06,h03v06,h03v07,h06v03,h07v03,h07v04,h07v05,h07v06,h07v07,h08v03,h08v04,h08v05,h08v06,h08v07," \
        "h09v02,h09v03,h09v04,h09v05,h09v06,h09v07,h10v02,h10v03,h10v04,h10v05,h10v06,h10v07,h11v02," \
        "h11v03,h11v04,h11v05,h11v06,h12v01,h12v02,h12v03,h12v04,h12v05,h13v01,h13v02,h13v03,h13v04," \
        "h14v01,h14v02,h14v03,h14v04,h15v01,h15v02,h15v03,h16v00,h16v01,h16v02,h17v00,h27v03,h28v03,h29v03".split(",")


def main(url, date):
    SUB_URL = url
    r_subfolder = requests.get(SUB_URL)
    soup_subfolder = BeautifulSoup(r_subfolder.text, "html.parser")

    date_folder = os.path.join("E:/Carbon_flux/data/regoin_image/mcd43a4", date)
    os.makedirs(date_folder, exist_ok=True)

    for tile in tiles:
        tile_found = False
        for link_carpetas in soup_subfolder.find_all("a"):
            file_name = link_carpetas.get("href")
            remote_file = SUB_URL + "/" + file_name

            if tile in remote_file and "hdf" in remote_file:
                dst_file = os.path.join(date_folder, file_name)

                if os.path.exists(dst_file) and os.path.getsize(dst_file) > 2048:
                    continue  # Skip if file exists and is not empty

                print(remote_file)
                s = requests.session()
                s.keep_alive = False

                cookies = {'_urs-gui_session': '642edb4284918efad3be3f265220bcd5'}
                requests.DEFAULT_RETRIES = 5  # Increase retry connection attempts

                response = requests.get(remote_file, cookies=cookies, timeout=300)

                with open(dst_file, "wb") as handle:
                    handle.write(response.content)
                    print("download_success", dst_file)

                tile_found = True
                break

        if not tile_found:
            print("Image for tile", tile, "does not exist")
            continue



if __name__ == '__main__':
    time_series = pd.date_range(start=datetime.strptime("2017-06-02", '%Y-%m-%d'),
                                end=datetime.strptime("2017-06-15", '%Y-%m-%d'), freq='d').to_list()
    url_list = [("https://e4ftl01.cr.usgs.gov/MOTA/MCD43A4.061/" + datetime.strftime(t, '%Y.%m.%d'),
                 datetime.strftime(t, '%Y-%m-%d')) for t in time_series]

    pool = Pool(60)
    pool.starmap(main, url_list)

# *****************************************************
# 下面这种方式是按照tile来存储所有日期的数据 也就是每个行列号所有天的数据存储在一起

# def main(SUB_URL):
#     # url para la descarga
#     r_subfolder = requests.get(SUB_URL)
#     # print(r_subfolder.text)
#     # parser información
#     soup_subfolder = BeautifulSoup(r_subfolder.text, "html.parser")
#
#     for link_carpetas in soup_subfolder.find_all("a"):
#         file_name = link_carpetas.get("href")
#         remote_file = SUB_URL + "/" + file_name
#
#         for tile in tiles:
#             if tile in remote_file and "hdf" in remote_file:
#                 dst_folder = "E:/Carbon_flux/data/regoin_image/mcd43a4"
#                 dst_folder = os.path.join(dst_folder, tile)
#                 os.makedirs(dst_folder, exist_ok=True)
#                 dst_file = os.path.join(dst_folder, file_name)
#
#                 if os.path.exists(dst_file) and os.path.getsize(dst_file) > 2048:
#                     continue  # Skip if file exists and is not empty
#                 print(remote_file)
#                 s = requests.session()
#                 s.keep_alive = False
#
#                 """
#                 非常重要！cookies _urs-gui_session 需要修改，有效期一般只有1天。文章末尾是修改方法
#                 https://zhuanlan.zhihu.com/p/585437107
#                 """
#                 cookies = {'_urs-gui_session': 'b82702f458ee90cb4146ed5ab7c37a2d'}
#                 requests.DEFAULT_RETRIES = 5  # 增加重试连接次数
#
#                 response = requests.get(remote_file, cookies=cookies, timeout=300)
#
#                 with open(dst_file, "wb") as handle:
#                     handle.write(response.content)
#                     print("download_success", dst_file)
#                 break
#
# if __name__ == '__main__':
#     # （需要修改）影像的起始和截止时间
#     time_series = pd.date_range(start=datetime.strptime("2017-01-01", '%Y-%m-%d'),
#                                 end=datetime.strptime("2017-01-03", '%Y-%m-%d'), freq='d').to_list()
#
#     # （需要修改）影像的下载地址，上图复制的链接
#     url_list = ["https://e4ftl01.cr.usgs.gov/MOTA/MCD43A4.061/" + datetime.strftime(t, '%Y.%m.%d') for t in time_series]
#     # print(url_list)
#     #
#     # for url in url_list:
#     #     main(url)
#     pool = Pool(20)  # （按需求修改）根据电脑配置设置Pool，数字越大越快，但是可能会爆内存
#     pool.map(main, url_list)
