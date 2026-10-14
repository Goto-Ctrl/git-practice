# 第5回 テスト計画（関数の切り出しとpytest）

- 学籍番号：2432032
- 氏名：後藤大地
- 提出日：10月14日
- 提出file名：`TEST_PLAN.md`（`~/practice` の直下に置き、commitして提出する）

> 各欄の例文や角括弧は、自分の実行結果と判断へ置き換えること。認証情報、password、tokenは記録しないこと。
> WSL2のUbuntu上で、pyenvの仮想環境を使って作業する。第4回でCopilotが生成した `~/practice/app.py` を題材にする。計算がすべて `main()` の中にある場合は、まず関数へ切り出してからtestを書く。

---

## 1. pytestの導入確認

- 使用したpyenvの仮想環境名：dla
- pytestを新たにinstallしたか：はい
- installした場合のcommand：python -m pip install pytest

```text
cd ~/git-practice
$ python --version
Python 3.12.11
$ pyenv version
pyenv 2.8.6
$ python -m pytest --version
pytest 9.1.1
```

- `requirements.txt` に `pytest` を追記したか：はい
- つまずいた点と対処：なし

---

## 2. 切り出し前のapp.pyの構造

関数の一覧と、計算を行っている行を確認する。

```text
$ grep -n "def " app.py
6:def calculate_channel_averages(image):
11:def calculate_channel_histograms(image):
```

| 行 |            code（要約）           |            計算 / 表示            |
|----|-----------------------------------|-----------------------------------|
| 6  |calculate_channel_averages(image)  |RGB各チャンネルの平均値を計算      |
| 11 |calculate_channel_histograms(image)|RGB各チャンネルのヒストグラムを計算|

- testから直接呼べる計算用の関数はあったか：
calculate_channel_averages(image) と calculate_channel_histograms(image) が
testから直接呼べる。

- 切り出し前のcommit ID：
e32c3bcb35f7056905e855b69f13e6fb7e6790f2

---

## 3. 関数の切り出し

### AIへ渡したprompt

```text
現在の app.py を確認してください。

次の2つの計算関数を analysis.py に切り出します。まず変更計画を立ててください。

calculate_channel_averages(image)
calculate_channel_histograms(image)
```

### AIの変更計画と人間の判断

- AIの計画（要約）：
計算処理の分離：2つの関数を analysis.py に移す。
既存アプリの維持：app.py から関数を削除してanalysis.pyからインポートし、
　　　　　　　　　画像解析・表示・エラー処理の挙動を変えない。
テストの作成と実行：平均値・配列形状・RGB各チャンネルのヒストグラムを検証する。

- 採用・修正した点と理由：
変更前後で同じ画像のRGB平均値とヒストグラムが一致することを確認するように修正した。

### 変更内容

```text
$ git diff -- app.py
 1 -   import numpy as np
 1     import streamlit as st
 2     from PIL import Image, UnidentifiedImageError
 3     
 5 -   
 6 -   def calculate_channel_averages(image):
 7 -       image_array = np.asarray(image)
 8 -       return image_array.shape, image_array.mean(axis=(0, 1))
 9 -   
10 -   
11 -   def calculate_channel_histograms(image):
12 -       image_array = np.asarray(image)
13 -       return {
14 -           "Red": np.histogram(image_array[:, :, 0], bins=256, range=(0, 256))[0],
15 -           "Green": np.histogram(image_array[:, :, 1], bins=256, range=(0, 256))[0],
16 -           "Blue": np.histogram(image_array[:, :, 2], bins=256, range=(0, 256))[0],
17 -       }
 4 +   from analysis import calculate_channel_averages, calculate_channel_histograms
 5     
 6     
 7     st.title("画像解析アプリ")
17             with Image.open(uploaded_image) as image:
18                 width, height = image.size
19                 rgb_image = image.convert("RGB")
33 -               array_shape, channel_averages = calculate_channel_averages(rgb_image)
34 -               st.write(f"幅: {width}px、高さ: {height}px")
35 -               st.write(f"array shape: {array_shape}")
36 -               st.write(
37 -                   "RGB channel average: "
38 -                   f"R={channel_averages[0]:.2f}, "
39 -                   f"G={channel_averages[1]:.2f}, "
40 -                   f"B={channel_averages[2]:.2f}"
41 -               )
42 -               st.subheader("RGB histogram")
43 -               st.line_chart(
44 -                   calculate_channel_histograms(rgb_image),
45 -                   color=["#FF0000", "#00AA00", "#0000FF"],
46 -                   x_label="画素値",
47 -                   y_label="画素数",
48 -               )
49 -               st.image(rgb_image, caption=uploaded_image.name)
20 +   
21 +           array_shape, channel_averages = calculate_channel_averages(rgb_image)
22 +           st.write(f"幅: {width}px、高さ: {height}px")
23 +           st.write(f"array shape: {array_shape}")
24 +           st.write(
25 +               "RGB channel average: "
26 +               f"R={channel_averages[0]:.2f}, "
27 +               f"G={channel_averages[1]:.2f}, "
28 +               f"B={channel_averages[2]:.2f}"
29 +           )
30 +           st.subheader("RGB histogram")
31 +           st.line_chart(
32 +               calculate_channel_histograms(rgb_image),
33 +               color=["#FF0000", "#00AA00", "#0000FF"],
34 +               x_label="画素値",
35 +               y_label="画素数",
36 +           )
37 +           st.image(rgb_image, caption=uploaded_image.name)
38         except (UnidentifiedImageError, OSError):
39             st.error("有効なJPEGまたはPNG画像を選択してください。")

```

- 新しく作ったfileと関数：

### 切り出し前後の確認

同じ画像を入力し、画面に表示されるRGB平均値が変わらないことを確認する。

| 項目 | 切り出し前 | 切り出し後 |
|---|---|---|
| 使用した画像 |red_1px|red_1px|
| R平均 |255|255|
| G平均 | 0 | 0 |
| B平均 | 0 | 0 |

- 切り出し後のcommit ID：
756a050eb1a5e877572cfc02eefda819911872e5
---

## 4. 対象関数と入力契約

- 対象関数：calculate_channel_averages(image)
- 定義場所：analysis.py
- 入力（typeとshape）：
PIL形式のRGB画像で、Numpy配列に変換した時のshapeは(H,W,3)となる。
Hは高さ、Wは幅、3はRGBの3チャンネルを表す。
- 処理：
入力画像をNumpy配列に変換し、RGB各チャンネルの平均値を求める。
- 出力（typeとshape）：
image_array.shapeは、画像の高さ、幅、チャンネル数をタプルで返す。
image_array.meanはRGB各チャンネルの平均値をNumpy配列で返す。

- 例外（契約違反の入力と発生させる例外）：
Noneを入力した場合、mean(axis=(0, 1)) の処理でAxisErrorが発生する。


```python
# 対象関数の定義をそのまま貼る
def calculate_channel_averages(image):
    image_array = np.asarray(image)
    return image_array.shape, image_array.mean(axis=(0, 1))
```

---

## 5. test case一覧

正常系2件以上、境界値2件以上、異常系1件以上。期待結果は手計算で求めた数値を書く。

| ID |  分類  | 入力                                                                           | 期待結果                                   | 選定理由                  |
| -- | ---    | -------------------------------------------------------------------------------| --------------------------------------     | ------------------------- |
| N1 | 正常系 | 2×2のRGB画像。画素値は `[[[10,20,30], [30,40,50]], [[50,60,70], [70,80,90]]]` | shape `(2,2,3)`、RGB平均値 `[40,50,60]`    | 複数画素のRGB平均値を正しく計算できるか確認 |
| N2 | 正常系 | 2×2のRGB画像。画素値は `[[[10,20,30], [10,20,30]], [[40,50,60], [40,50,60]]]` | shape `(2,2,3)`、RGB平均値 `[25,35,45]`    | 複数画素のヒストグラムを正しく作れるか検証     |
| B1 | 境界値 | 1×1の黒画像 `(0,0,0)`                                                         | shape `(1,1,3)`、RGB平均値 `[0,0,0]`       | 最小画素値と1画素でヒストグラムを確認      |
| B2 | 境界値 | 1×1の白画像 `(255,255,255)`                                                   | shape `(1,1,3)`、RGB平均値 `[255,255,255]` | 最大画素値と1画素でヒストグラムを確認      |
| E1 | 異常系 | `None`                                                                         | 例外が発生することを想定。                 | 想定していない入力に対する挙動を確認 |

---

## 6. pytest実装

対象関数と同じdirectoryの `test_analysis.py` に書いたtest関数を貼る。

```python
# 通常系1：画像の形状とRGB平均値をテスト
def test_normal_rgb_averages():
    pixels = np.array(
        [
            [[10, 20, 30], [30, 40, 50]],
            [[50, 60, 70], [70, 80, 90]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    shape, averages = calculate_channel_averages(image)

    assert shape == (2, 2, 3)
    np.testing.assert_allclose(averages, [40, 50, 60])


# 通常系2：RGBヒストグラムをテスト
def test_normal_rgb_histograms():
    pixels = np.array(
        [
            [[10, 20, 30], [10, 20, 30]],
            [[40, 50, 60], [40, 50, 60]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(pixels)

    histograms = calculate_channel_histograms(image)

    assert set(histograms.keys()) == {"Red", "Green", "Blue"}

    expected_values = {
        "Red": [10, 40],
        "Green": [20, 50],
        "Blue": [30, 60],
    }

    for channel, values in expected_values.items():
        expected = np.zeros(256, dtype=np.int64)

        for value in values:
            expected[value] = 2

        np.testing.assert_array_equal(histograms[channel], expected)


# 境界値1：黒（画素値0）の1ピクセル画像
def test_boundary_black_pixel():
    image = Image.new("RGB", (1, 1), (0, 0, 0))

    shape, averages = calculate_channel_averages(image)

    assert shape == (1, 1, 3)
    np.testing.assert_allclose(averages, [0, 0, 0])


# 境界値2：白（画素値255）の1ピクセル画像
def test_boundary_white_pixel():
    image = Image.new("RGB", (1, 1), (255, 255, 255))

    shape, averages = calculate_channel_averages(image)
    histograms = calculate_channel_histograms(image)

    assert shape == (1, 1, 3)
    np.testing.assert_allclose(averages, [255, 255, 255])

    for channel in ("Red", "Green", "Blue"):
        assert histograms[channel][255] == 1
        assert histograms[channel].sum() == 1


# 異常系：Noneを渡した場合の例外をテスト
def test_invalid_input_for_histograms():
    with pytest.raises(IndexError):
        calculate_channel_histograms(None)
```

---

## 7. pytest実行結果

最初の実行でFAILがあった場合は、その出力と対処も残す。

```text
$ pytest -q
.....                                                                           [100%]
5 passed in 0.33s
```

- FAILの有無と対処（例：異常系が `DID NOT RAISE` となり、入力契約の確認を追加した）：
実行した結果、FAILは発生しなかったので、対処はしなかった。
正常系2件、境界値2件、異常系1件の計5件すべてがPASSEDとなり、期待した結果が得られることを確認した。
---

## 8. 意図的bugを検出した例

- 入れたbug：
channel順を `[B, G, R]` に変更

- 失敗したtestと出力：

```text
FAILED test_analysis.py::test_normal_rgb_averages - AssertionError:
1 failed, 4 passed in 0.41s
```

test_normal_rgb_averagesでは、赤と青のチャンネルの順番が入れ替わったので、
想定される結果と出力が異なり、テストが失敗した。従って、RGBチャンネルの
順番の誤りを検出できることが分かった。

- bugを戻した後の結果：

```text
.....                                                                                                                                                                     [100%]
5 passed in 0.61s
```
RGBチャンネルの順番を元どおりにした結果、

---

## 9. AI提案の採否

| AIが提案した内容（切り出し・test case） | 採用 / 不採用 | 理由 |
|---|---|---|
|黒画像と白画像を使った境界値テストを追加する| 採用 | 画素値の最小値と最大値について、平均値が正しく計算されるか確認できるため。|

---

## 提出前の確認

- [x] `pyenv version` と `pytest --version` の結果を書いた
- [x] 切り出し前のapp.pyで、計算と表示の行を区別した
- [x] 切り出しのpromptと `git diff` を貼った
- [x] 切り出し前後で同じ画像のRGB平均値が一致することを確認した
- [x] 対象関数の入力契約（type・shape・例外）を書いた
- [x] 正常系2件・境界値2件・異常系1件以上を表にし、期待結果を手計算で示した
- [x] pytestの実装と実行結果を貼った
- [x] 意図的bugで失敗する様子と戻した結果を貼った
- [x] AI提案の採否と理由を書いた
- [x] `~/git-practice` 直下に `TEST_PLAN.md` として保存し、commitした
