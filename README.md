# Alpino Doors — qapı vizualizasiyası

Boşluğun şəkli + ölçü (mm) + qapı modeli → qapının həmin boşluqda necə görünəcəyini göstərən real şəkil.
Hər şey öz serverinizdə işləyir, xarici servis yoxdur.

```
şəkil ──► Model 1 (boşluq, 4 künc) ──► perspektiv + mm ölçü ──► qapı oturdulur ──► işıq/kölgə ──► nəticə
                                                  ▲
           kataloq: parametrik modellər  ─────────┤
           və ya Model 2 ilə şəkildən kəsilmiş qapı
```

## İki model

| | Nə edir | Tip | Etiket |
|---|---|---|---|
| **Model 1** `weights/opening.pt` | Şəkildə qapı boşluğunu və 4 küncünü tapır | YOLO11-pose (4 keypoint) | Hər boşluğa 4 nöqtəli poliqon |
| **Model 2** `weights/door.pt` | Instagram/kataloq şəklindən qapını kəsir → şəffaf fonlu şablon | YOLO11-seg | Qapının konturu (poliqon) |

Hər ikisi kiçik modeldir, transfer learning ilə öyrədilir. 150–300 şəkil kifayət edir, GPU-da 15–30 dəqiqə çəkir.
Modellər hələ öyrədilməyibsə, sistem yenə işləyir: künclər interfeysdə əl ilə qoyulur.

## Quraşdırma

```bash
git clone https://github.com/elnurbx-work/alpinoDoors && cd alpinoDoors
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.server:app --host 0.0.0.0 --port 8000
# brauzerdə: http://SERVER_IP:8000
```

Docker ilə:

```bash
docker build -t alpino-doors .
docker run -d -p 8000:8000 -v $PWD/weights:/app/weights -v $PWD/catalog:/app/catalog alpino-doors
```

## Öyrətmə (train)

### 0. Saytdan dataset yığmaq (ən asan yol)
Saytda hər «Vizualı hazırla» basılanda şəkil və künclər avtomatik olaraq hazır etiket kimi `dataset/` qovluğuna yazılır (`ALPINO_DATASET`). Yuxarıdakı «Dataset» sayğacı neçə şəkil yığıldığını göstərir.
Yükləmək üçün: `https://SAYT/api/dataset/export?token=ADMIN_TOKEN` → zip, sonra:

```bash
unzip alpino_dataset_*.zip -d export_sayt
python scripts/prepare_dataset.py --task opening --src export_sayt --out data/opening
python scripts/train.py opening
```

### 1. Şəkilləri toplayın
- **Model 1:** obyektlərdən boşluq şəkilləri (WhatsApp-dan gələnlər idealdır), müxtəlif bucaq və işıqda. Hədəf: 200+ şəkil.
- **Model 2:** qapılarınızın şəkilləri (Instagram, kataloq). Hədəf: 150+ şəkil.

### 1b. Açıq dataset (ilkin öyrətmə)
`weights/opening_pretrain.pt` — [DoorDetect](https://github.com/MiguelARD/DoorDetect-Dataset) (Open Images + MCIndoor, 335 qapılı + 150 fon şəkli) üzərində öyrədilib.
Təkbaşına təmir şəkillərində zəifdir (hazır qapıları tanıyır, boş boşluqları yox), ona görə saytda istifadə olunmur —
öz datanızla öyrədərkən başlanğıc kimi verin: `python scripts/train.py opening --base weights/opening_pretrain.pt`.

### 2. Etiketləyin — avtomatik (tövsiyə olunur)
Model 1 üçün əl ilə etiket yazmaq lazım deyil: açıq lüğətli YOLOE + SAM 2 boşluqları özü tapır, sonra yoxlama vərəqlərinə baxıb pisləri atırsınız.

```bash
pip install "git+https://github.com/ultralytics/CLIP.git"
python scripts/auto_label.py --src bosluq_sekilleri/ --out data/opening
# data/opening/review/sheet_*.jpg vərəqlərinə baxın, pisləri rejected.txt-yə yazın (şəkil adı və ya ad:indeks)
python scripts/auto_label.py --src bosluq_sekilleri/ --out data/opening --reject rejected.txt
```

### 2b. Etiketləyin — əl ilə
[CVAT](https://www.cvat.ai), [Label Studio](https://labelstud.io) və ya [Roboflow](https://roboflow.com) istifadə edin.
- **Model 1:** hər boşluğu **4 nöqtəli poliqon** ilə çəkin. Künclər divarın üz tərəfində olmalıdır, qapının qoyulacağı yerdə. Class: `opening`.
- **Model 2:** qapının konturunu poliqonla çəkin. Class: `door`.
- Export: **YOLO segmentation** formatı (`images/` + `labels/`).

### 3. Dataseti hazırlayın və öyrədin

```bash
python scripts/prepare_dataset.py --task opening --src export_bosluq --out data/opening
python scripts/prepare_dataset.py --task door    --src export_qapi   --out data/door

python scripts/train.py opening --device 0      # GPU yoxdursa: --device cpu
python scripts/train.py door    --device 0
```

Model 2-ni əlavə olaraq sintetik datasetlə gücləndirmək olar: parametrik qapılar boşluq şəkillərinin üzərinə avtomatik qoyulur və etiketlər özü yazılır.

```bash
python scripts/synth_doors.py --bg bosluq_sekilleri/ --out data/door_synth --n 500
```

## İstifadə

**Veb interfeys:** şəkli yükləyin → ölçünü yazın → modeli seçin → «Vizualı hazırla». Telefondan da işləyir.

**Komanda sətri:**

```bash
python scripts/render.py --photo bosluq.jpg --width 1588 --height 2518 --door french_arch --annotate
python scripts/render.py --photo bosluq.jpg --width 1770 --height 2160 --door double_bar \
       --quad "226,292 780,318 770,1140 245,1213"
```

**Kataloqa yeni qapı əlavə etmək (Model 2):**

```bash
python scripts/make_template.py --photo insta_post.jpg --id cut_qara_01 --name "Cüt tay, qara" --width 1600 --height 2400
```

Qapı bağlı və düz qarşıdan çəkilmiş olmalıdır. Qara profil saxlanılır, şüşə yarı-şəffaf olur ki, yeni otaq şüşədən görünsün.

## Parametrik modellər

`alpino/parametric.py` faylında. Qapı dəqiq mm ölçüsü ilə çəkilir. Profil eni, tay sayı, tağ, arakəsmələr, şüşə növü (şəffaf / bronz / rifli / mat) və tutacaq dəyişdirilə bilər.
Yeni model əlavə etmək üçün `PRESETS` lüğətinə bir sətir yazmaq kifayətdir.

## API

| Endpoint | |
|---|---|
| `GET /api/doors` | Model siyahısı |
| `POST /api/detect` | `photo` → `{quad, auto, size}` |
| `POST /api/render` | `photo, quad, door_id, width_mm, height_mm, [inset_mm, glass, annotate]` → JPEG |
| `POST /api/template` | `photo, name, [quad, width_mm, height_mm]` → kataloqa yeni qapı |
