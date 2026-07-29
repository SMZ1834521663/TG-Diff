import os
import re
import time
import requests
import json

BASE_URL = "https://cad.onshape.com"
ACCESS_KEY = "on_mOTVsZIOoiB5GmCO0ztjx"
SECRET_KEY = "rDpZou3TVqpzWpGxk0Z4NgpVujyiqb0JZ05FYEGwBHX6RBkK"

type="train"
json_path = "/data/ybc2021/Datasets/Furniture/{}.json".format(type)
with open(json_path, "r", encoding="utf-8") as f:
    data_list = json.load(f)

print(len(data_list))
json_headers = {
    "Accept": "application/json;charset=UTF-8; qs=0.09",
    "Content-Type": "application/json;charset=UTF-8; qs=0.09",
}

def download_step(item):
    did, wid, eid = item["did"], item["wid"], item["eid"]

    # 1) Initiate asynchronous STEP export
    export_url = f"{BASE_URL}/api/v11/partstudios/d/{did}/w/{wid}/e/{eid}/export/step"
    r = requests.post(
        export_url,
        auth=(ACCESS_KEY, SECRET_KEY),
        headers=json_headers,
        json={"storeInDocument": False},
        timeout=30,
    )
    r.raise_for_status()
    translation_id = r.json()["id"]

    # 2) Polling export status
    poll_url = f"{BASE_URL}/api/v9/translations/{translation_id}"
    fid = None
    for _ in range(120):
        s = requests.get(
            poll_url,
            auth=(ACCESS_KEY, SECRET_KEY),
            headers={"Accept": "application/json;charset=UTF-8; qs=0.09"},
            timeout=30,
        )
        s.raise_for_status()
        info = s.json()
        state = info.get("requestState")

        if state == "DONE":
            fids = info.get("resultExternalDataIds") or []
            if not fids:
                raise RuntimeError(f"Export completed but no resultExternalDataIds: {info}")
            fid = fids[0]
            break
        if state == "FAILED":
            raise RuntimeError(f"error: {info.get('failureReason')}")
        time.sleep(4)

    if not fid:
        raise TimeoutError("over time")

    # 3) Download external data files
    dl_url = f"{BASE_URL}/api/v6/documents/d/{did}/externaldata/{fid}"
    f = requests.get(
        dl_url,
        auth=(ACCESS_KEY, SECRET_KEY),
        headers={"Accept": "application/octet-stream"},
        timeout=120,
    )
    f.raise_for_status()

    # Prioritize using response header file names
    cd = f.headers.get("Content-Disposition", "")
    m = re.search(r'filename="?([^"]+)"?', cd)
    filename = m.group(1) if m else f'{item["data_id"]}_{item["category"]}.step'

    with open("/data/ybc2021/Datasets/Furniture/{}".format(type)+"/"+filename, "wb") as out:
        out.write(f.content)

    print(f"download: {filename}")

if __name__ == "__main__":
    folder = "/data/ybc2021/Datasets/Furniture/{}".format(type)
    all_saved = os.listdir(folder)
    for row in data_list:
        try:
            if row["data_id"] + "_" + row["category"] + ".step" in all_saved:
                continue
            download_step(row)
        except:
            print(row)
            pass