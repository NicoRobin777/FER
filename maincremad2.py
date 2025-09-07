import os, re, cv2, numpy as np, librosa, joblib, tensorflow as tf
import shutil, subprocess, tempfile
from collections import Counter

from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.layers import (
    Input, Dense, Conv2D, MaxPooling2D, Dropout, BatchNormalization,
    GlobalAveragePooling2D, Concatenate, TimeDistributed, LayerNormalization,
    MultiHeadAttention, Layer
)
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt, seaborn as sns
from tqdm import tqdm
from mtcnn import MTCNN
from tensorflow.keras import mixed_precision
from keras.saving import register_keras_serializable
import warnings
warnings.filterwarnings('ignore')

print("TensorFlow version:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices('GPU'))
for g in tf.config.list_physical_devices('GPU'):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception as e:
        print("set_memory_growth failed:", e)
mixed_precision.set_global_policy("mixed_float16")

DATASET_PATH = "/mnt/p/NS/CREMA-D/VideoFlash"
PREPROCESSED_PATH = "/mnt/p/NS/CREMA-D/preprocessed_cremad_data_SPECTRO_v1.npz"
MODEL_SAVE_PATH = "/mnt/p/NS/CREMA-D/models/cremad_multimodal_SPECTRO_v1.keras"
LABEL_ENCODER_PATH = "/mnt/p/NS/CREMA-D/label_encoder.joblib"

def resolve_dataset_path(p):
    """Map Windows-style paths (e.g., P:\folder) to WSL (/mnt/p/folder) when needed."""
    if os.path.exists(p):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest  = m.group(2).replace("\\", "/")
        candidate = f"/mnt/{drive}/{rest}"
        if os.path.exists(candidate):
            print(f"[WSL path mapped] {p} -> {candidate}")
            return candidate
    return p

def resolve_file_path(p):
    """Same idea but for a single file path."""
    if os.path.exists(p):
        return p
    m = re.match(r"^([A-Za-z]):[\\/](.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest  = m.group(2).replace("\\", "/")
        candidate = f"/mnt/{drive}/{rest}"
        if os.path.exists(candidate):
            print(f"[WSL path mapped] {p} -> {candidate}")
            return candidate
    return p

def np_no_nan(x): return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
def safe_remove(p):
    try:
        if os.path.exists(p): os.remove(p)
    except: pass

def resolve_ffmpeg_exe():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe): return exe
    except Exception:
        pass
    return None

#Positional Encoding 
@register_keras_serializable(package="custom", name="SinusoidalPE")
class SinusoidalPE(Layer):
    """Sinusoidal positional encoding for (B,T,D). Mixed-precision safe and serializable."""
    def call(self, x):
        T = tf.shape(x)[1]
        D = tf.shape(x)[2]
        pos = tf.cast(tf.range(T)[:, tf.newaxis], tf.float32)   
        i   = tf.cast(tf.range(D)[tf.newaxis, :], tf.float32)   
        angle_rates = 1.0 / tf.pow(10000.0, (tf.floor(i/2.0) * 2.0) / tf.cast(D, tf.float32))
        angle_rads  = pos * angle_rates
        sines   = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pe      = tf.concat([sines, cosines], axis=-1)          
        peD = tf.shape(pe)[-1]
        pe  = tf.cond(peD < D, lambda: tf.pad(pe, [[0,0],[0, D - peD]]), lambda: pe[:, :D])
        pe  = tf.cast(pe[tf.newaxis, ...], x.dtype)             
        return x + pe
    def compute_output_shape(self, input_shape):
        return input_shape
    def get_config(self):
        return super().get_config()

def transformer_encoder(inputs, num_heads=4, ff_dim=128, dropout_rate=0.1):
    """Encoder block that avoids raw tf ops on KerasTensor."""
    x = SinusoidalPE()(inputs)
    D_static = inputs.shape[-1]
    D_use = int(D_static) if D_static is not None else 128
    key_dim = max(8, D_use // num_heads)
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=key_dim)(x, x)
    attn = Dropout(dropout_rate)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(x + attn)
    ffn = Dense(ff_dim, activation="relu")(out1)
    ffn = Dense(D_use)(ffn)
    ffn = Dropout(dropout_rate)(ffn)
    return LayerNormalization(epsilon=1e-6)(out1 + ffn)

class RAVDESSEmotionRecognizer:
    def __init__(self, dataset_path, max_frames=30, face_size=64, use_mtcnn=True, n_mels=128, spec_time=128):
        self.dataset_path = dataset_path
        self.max_frames, self.face_size = max_frames, face_size
        self.use_mtcnn, self.n_mels, self.spec_time = use_mtcnn, n_mels, spec_time
        self.detector = None
        if use_mtcnn:
            _orig_policy = mixed_precision.global_policy()
            try:
                mixed_precision.set_global_policy("float32")
            except Exception:
                pass
            try:
                with tf.device("/CPU:0"):
                    self.detector = MTCNN()
            finally:
                try:
                    mixed_precision.set_global_policy(_orig_policy.name)
                except Exception:
                    pass
        self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.label_encoder = LabelEncoder()
        self.emotions = {'ANG':'angry','DIS':'disgust','FEA':'fearful','HAP':'happy','NEU':'neutral','SAD':'sad'}

    def parse_filename(self, fname):
        base = os.path.basename(fname)
        parts = base.split('_')
        return self.emotions.get(parts[2], 'unknown') if len(parts) >= 3 else 'unknown'

#video
    def _largest_face_box(self, frame_bgr):
        if self.use_mtcnn and self.detector is not None:
            with tf.device("/CPU:0"):
                res = self.detector.detect_faces(frame_bgr)
            if res:
                x,y,w,h = max((r['box'] for r in res), key=lambda b: b[2]*b[3]); return max(0,x), max(0,y), w, h
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        det = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
        if len(det): x,y,w,h = max(det, key=lambda b: b[2]*b[3]); return x,y,w,h
        return None

    def extract_faces_from_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or self.max_frames
        idxs = np.linspace(0, max(0,total-1), num=self.max_frames, dtype=int)
        faces=[]
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret or frame is None:
                faces.append(np.zeros((self.face_size,self.face_size), np.uint8)); continue
            box = self._largest_face_box(frame)
            if box is not None:
                x,y,w,h = box; y2,x2 = y+h, x+w
                y,x = max(0,y), max(0,x); y2,x2 = min(frame.shape[0],y2), min(frame.shape[1],x2)
                face = frame[y:y2, x:x2]
                if face.size == 0:
                    faces.append(np.zeros((self.face_size,self.face_size), np.uint8))
                else:
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (self.face_size,self.face_size))
                    faces.append(face)
            else:
                faces.append(np.zeros((self.face_size,self.face_size), np.uint8))
        cap.release()
        if len(faces) == 0:
            faces = [np.zeros((self.face_size,self.face_size), np.uint8) for _ in range(self.max_frames)]
        faces = np.array(faces, dtype=np.uint8)
        if faces.shape[0] < self.max_frames:
            pad = self.max_frames - faces.shape[0]
            faces = np.concatenate([faces, np.zeros((pad, self.face_size, self.face_size), np.uint8)], axis=0)
        elif faces.shape[0] > self.max_frames:
            faces = faces[:self.max_frames]
        return np.array(faces)

#audio (spectrogram) 
    def extract_audio_spectrogram(self, video_path, sr=22050, n_fft=1024, hop_length=512):
        tmp_wav = None
        try:
            ffmpeg = resolve_ffmpeg_exe()
            if not ffmpeg:
                raise RuntimeError("FFmpeg not found. Add ffmpeg to PATH or pip install imageio-ffmpeg.")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpf:
                tmp_wav = tmpf.name
            cmd = [ffmpeg, "-i", video_path, "-vn", "-acodec", "pcm_s16le",
                   "-ar", str(sr), "-ac", "1", "-y", tmp_wav]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0 or (not os.path.exists(tmp_wav) or os.path.getsize(tmp_wav) == 0):
                err = proc.stderr.decode(errors='ignore')
                raise RuntimeError(f"ffmpeg failed for {video_path}:\n{err[:800]}")

            y, sr_loaded = librosa.load(tmp_wav, sr=sr, mono=True)
            if y.size == 0:
                raise RuntimeError("Extracted audio is empty.")
            S = librosa.feature.melspectrogram(y=y, sr=sr_loaded, n_mels=self.n_mels,
                                               n_fft=n_fft, hop_length=hop_length, power=2.0)
            S_db = librosa.power_to_db(S, ref=np.max)
            if S_db.shape[1] < self.spec_time:
                S_db = librosa.util.fix_length(S_db, size=self.spec_time, axis=1)
            else:
                S_db = S_db[:, :self.spec_time]
            S_db = np_no_nan(S_db)
            S_min, S_max = float(S_db.min()), float(S_db.max())
            S_norm = (S_db - S_min) / (S_max - S_min + 1e-6)
            return S_norm[..., np.newaxis].astype(np.float32)
        except Exception as e:
            print(f"[AudioSpec] {video_path}: {e}")
            return np.zeros((self.n_mels, self.spec_time, 1), dtype=np.float32)
        finally:
            try:
                if tmp_wav and os.path.exists(tmp_wav):
                    os.remove(tmp_wav)
            except Exception:
                pass

    def build_dataset_from_scratch(self):
        v_feats, a_specs, labels = [], [], []
        files=[]
        for root,_,fs in os.walk(self.dataset_path):
            for f in fs:
                if f.lower().endswith(('.mp4','.flv','.avi','.mov','.mkv')):
                    files.append((f, os.path.join(root,f)))
        print(f"Found {len(files)} video files")

        kept = 0
        for fname, path in tqdm(files, desc="Processing videos"):
            emo = self.parse_filename(fname)
            if emo=='unknown': 
                continue
            faces = self.extract_faces_from_video(path)
            spec  = self.extract_audio_spectrogram(path)
            v_feats.append(faces); a_specs.append(spec); labels.append(emo)
            kept += 1

        print(f"Kept {kept} labeled samples.")
        if not labels:
            raise RuntimeError(
                "No labeled samples were found. Check DATASET_PATH and filename parsing. Expected like 1001_DFA_ANG_XX.flv"
            )
        print("Label distribution:", Counter(labels))

        y_enc = self.label_encoder.fit_transform(labels)
        joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)

        Xv = np.array([np.expand_dims(f.astype(np.float32)/255.0, -1) for f in v_feats], dtype=np.float32)  
        Xa = np.array(a_specs, dtype=np.float32)                                                             

        counts = np.bincount(y_enc)
        strat = y_enc if counts.min() >= 2 else None
        if strat is None:
            print("Warning: Not enough samples per class for stratified split; using random split.")

        Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te = train_test_split(
            Xv, Xa, y_enc, test_size=0.2, random_state=42, stratify=strat
        )
        y_tr_oh = tf.keras.utils.to_categorical(y_tr, len(self.label_encoder.classes_))
        y_te_oh = tf.keras.utils.to_categorical(y_te, len(self.label_encoder.classes_))

        os.makedirs(os.path.dirname(PREPROCESSED_PATH), exist_ok=True)
        np.savez_compressed(
            PREPROCESSED_PATH,
            X_video_train=Xv_tr, X_video_test=Xv_te,
            X_audio_train=Xa_tr, X_audio_test=Xa_te,
            y_train=y_tr_oh, y_test=y_te_oh,
            classes=np.array(self.label_encoder.classes_, dtype=str),
            spec_shape=np.array([self.n_mels, self.spec_time], dtype=np.int32)
        )
        return Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr_oh, y_te_oh

    def load_cached_dataset(self):
        if not os.path.exists(PREPROCESSED_PATH): return None
        try:
            with np.load(PREPROCESSED_PATH, allow_pickle=True) as d:
                Xv_tr, Xv_te = d['X_video_train'], d['X_video_test']
                Xa_tr, Xa_te = d['X_audio_train'], d['X_audio_test']
                y_tr,  y_te  = d['y_train'], d['y_test']
                classes = d['classes'] if 'classes' in d.files else None
            if Xa_tr.ndim != 4 or Xa_te.ndim != 4 or Xv_tr.ndim != 5:
                print("Cached data has wrong shapes; rebuilding...")
                return None
            try:
                self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            except Exception:
                if classes is not None: self.label_encoder.classes_ = classes
                joblib.dump(self.label_encoder, LABEL_ENCODER_PATH)
            return Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None

#model 
    def build_model(self, video_shape, audio_shape, num_classes):
        #video
        v_in = Input(shape=video_shape, name='video_input')
        x = TimeDistributed(Conv2D(32,3,activation='relu',padding='same'))(v_in)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(2))(x)
        x = TimeDistributed(Conv2D(64,3,activation='relu',padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(MaxPooling2D(2))(x)
        x = TimeDistributed(Conv2D(128,3,activation='relu',padding='same'))(x)
        x = TimeDistributed(BatchNormalization())(x)
        x = TimeDistributed(GlobalAveragePooling2D())(x)
        x = Dense(128)(x)
        x = transformer_encoder(x, num_heads=4, ff_dim=256, dropout_rate=0.1)
        v_feat = tf.keras.layers.GlobalAveragePooling1D()(x)

        #audio (spectrogram CNN)
        a_in = Input(shape=audio_shape, name='audio_input')  # (n_mels, spec_time, 1)
        a = Conv2D(32,3,activation='relu',padding='same')(a_in); a = MaxPooling2D(2)(a); a = BatchNormalization()(a)
        a = Conv2D(64,3,activation='relu',padding='same')(a);    a = MaxPooling2D(2)(a); a = BatchNormalization()(a)
        a = Conv2D(128,3,activation='relu',padding='same')(a);   a = MaxPooling2D(2)(a); a = BatchNormalization()(a)
        a = Conv2D(256,3,activation='relu',padding='same')(a);   a = GlobalAveragePooling2D()(a); a = Dropout(0.5)(a)
        a_feat = Dense(128, activation='relu')(a)

        z = Concatenate()([v_feat, a_feat])
        z = Dense(128, activation='relu')(z); z = Dropout(0.5)(z)
        z = Dense(64, activation='relu')(z);  z = Dropout(0.5)(z)
        out = Dense(num_classes, activation='softmax', dtype="float32")(z)  # float32 output under mixed precision
        model = Model([v_in, a_in], out)

        try:
            from tensorflow.keras.optimizers import AdamW
        except Exception:
            from tensorflow.keras.optimizers.experimental import AdamW
        model.compile(
            optimizer=AdamW(learning_rate=3e-4, weight_decay=1e-4),
            loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
            metrics=['accuracy'],
            jit_compile=False  
        )
        return model

#Training/Evaluation
    def train(self):
        loaded = self.load_cached_dataset()
        if loaded is None:
            print("Building dataset with spectrograms...")
            Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te = self.build_dataset_from_scratch()
        else:
            print("Loaded cached spectrogram dataset.")
            Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te = loaded

        Xv_tr = Xv_tr.astype(np.float16); Xv_te = Xv_te.astype(np.float16)
        Xa_tr = Xa_tr.astype(np.float16); Xa_te = Xa_te.astype(np.float16)

        num_classes = y_tr.shape[1]
        model = self.build_model(Xv_tr.shape[1:], Xa_tr.shape[1:], num_classes)

        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)
        ]
        y_tr_int = np.argmax(y_tr, axis=1)
        weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=y_tr_int)
        class_weight = {i: float(w) for i, w in enumerate(weights)}
        print("Class weights:", class_weight)

        AUTOTUNE = tf.data.AUTOTUNE
        def make_ds(Xv, Xa, y, batch=8, shuffle=True):
            ds = tf.data.Dataset.from_tensor_slices(({"video_input": Xv, "audio_input": Xa}, y))
            if shuffle:
                ds = ds.shuffle(min(len(Xv), 1000), reshuffle_each_iteration=True)
            return ds.batch(batch).prefetch(AUTOTUNE)

        train_ds = make_ds(Xv_tr, Xa_tr, y_tr, batch=8, shuffle=True)
        val_ds   = make_ds(Xv_te, Xa_te, y_te, batch=8, shuffle=False)

        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=100,
            callbacks=cbs,
            class_weight=class_weight,
            verbose=1
        )

        tl, ta = model.evaluate(val_ds, verbose=0)
        print(f"\nTest Accuracy: {ta:.4f}")

        y_pred = model.predict(val_ds, verbose=0)
        yp = np.argmax(y_pred, axis=1)
        yt = np.argmax(y_te, axis=1)
        print("\nClassification Report:")
        print(classification_report(yt, yp, target_names=list(self.label_encoder.classes_)))

        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
        model.save(MODEL_SAVE_PATH); print(f"Model saved: {MODEL_SAVE_PATH}")

        self.plot_history(history); self.plot_cm(yt, yp)
        return model

    def evaluate_saved(self):
        if not os.path.exists(PREPROCESSED_PATH):
            print("No preprocessed data to evaluate."); return
        loaded = self.load_cached_dataset()
        if loaded is None:
            print("Cached data invalid; please run training once to rebuild."); return
        Xv_tr, Xv_te, Xa_tr, Xa_te, y_tr, y_te = loaded

        Xv_te = Xv_te.astype(np.float16); Xa_te = Xa_te.astype(np.float16)

        model = tf.keras.models.load_model(
            MODEL_SAVE_PATH,
            compile=False,
            custom_objects={"SinusoidalPE": SinusoidalPE}
        )

        AUTOTUNE = tf.data.AUTOTUNE
        val_ds = tf.data.Dataset.from_tensor_slices(({"video_input": Xv_te, "audio_input": Xa_te}, y_te)).batch(8).prefetch(AUTOTUNE)

        y_pred = model.predict(val_ds, verbose=0)
        yp, yt = np.argmax(y_pred,1), np.argmax(y_te,1)
        acc = accuracy_score(yt, yp)
        print(f"Overall Test Accuracy: {acc:.4f} ({acc*100:.2f}%)\n")
        print(classification_report(yt, yp, target_names=list(self.label_encoder.classes_)))

        f1s = f1_score(yt, yp, average=None); names = list(self.label_encoder.classes_)
        plt.figure(figsize=(7,4)); plt.bar(range(len(names)), f1s)
        plt.xticks(range(len(names)), names, rotation=45, ha='right'); plt.ylabel('F1'); plt.title('Per-class F1')
        plt.tight_layout(); plt.savefig('fig_per_class_f1.png', dpi=300); plt.show()

        self.plot_cm(yt, yp)

    def predict_one(self, video_path):
        model = tf.keras.models.load_model(
            MODEL_SAVE_PATH, compile=False, custom_objects={"SinusoidalPE": SinusoidalPE}
        )
        try:
            self.label_encoder = joblib.load(LABEL_ENCODER_PATH)
            classes = list(self.label_encoder.classes_)
        except Exception:
            classes = ['angry','disgust','fearful','happy','neutral','sad']

        video_path = resolve_file_path(video_path)
        faces = self.extract_faces_from_video(video_path)
        spec  = self.extract_audio_spectrogram(video_path)
        if faces is None or len(faces)==0: return "No frames extracted / face not detected."

        faces = np.expand_dims(np.expand_dims(faces.astype(np.float32)/255.0, -1), 0)  
        spec  = np.expand_dims(spec, 0)                                                

        p = model.predict([faces, spec], verbose=0)
        k = int(np.argmax(p,1)[0]); conf = float(np.max(p))
        return {'emotion': classes[k], 'confidence': conf,
                'all_probabilities': {classes[i]: float(p[0,i]) for i in range(len(classes))}}

#Plotting
    def plot_history(self, h):
        fig,(ax1,ax2)=plt.subplots(1,2,figsize=(15,5))
        ax1.plot(h.history['accuracy']); ax1.plot(h.history['val_accuracy']); ax1.set_title('Accuracy'); ax1.legend(['train','val']); ax1.grid(True)
        ax2.plot(h.history['loss']); ax2.plot(h.history['val_loss']); ax2.set_title('Loss'); ax2.legend(['train','val']); ax2.grid(True)
        plt.tight_layout(); plt.savefig('training_history.png', dpi=300); plt.show()

    def plot_cm(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_, yticklabels=self.label_encoder.classes_)
        plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('Actual')
        plt.tight_layout(); plt.savefig('confusion_matrix.png', dpi=300); plt.show()


if __name__ == "__main__":
    try:
        choice = int(input("0=train, 1=test one video, 2=evaluate saved model: "))
    except Exception:
        choice = 0

    ds_path = resolve_dataset_path(DATASET_PATH)
    preproc_path = resolve_file_path(PREPROCESSED_PATH)
    model_path = resolve_file_path(MODEL_SAVE_PATH)
    le_path = resolve_file_path(LABEL_ENCODER_PATH)

    DATASET_PATH = ds_path
    PREPROCESSED_PATH = preproc_path
    MODEL_SAVE_PATH = model_path
    LABEL_ENCODER_PATH = le_path

    try:
        import glob
        exts = ("*.mp4","*.flv","*.avi","*.mov","*.mkv")
        some = []
        for e in exts:
            some += glob.glob(os.path.join(ds_path, "**", e), recursive=True)
            if len(some) >= 5: break
        print(f"[Dataset sanity] example files ({len(some)} shown):")
        for s in some[:5]: print("  ", s)
    except Exception:
        pass

    recognizer = RAVDESSEmotionRecognizer(
        ds_path, max_frames=30, face_size=64, use_mtcnn=True, n_mels=128, spec_time=128
    )

    if choice == 0:
        recognizer.train()
    elif choice == 1:
        default_demo = resolve_file_path("/mnt/p/NS/CREMA-D/VideoFlash/VideoFlash/1001_DFA_FEA_XX.flv")
        path = input("Path to test video (default demo): ").strip() or default_demo
        res = recognizer.predict_one(path)
        if isinstance(res, dict):
            print(f"Predicted: {res['emotion']} (conf {res['confidence']:.3f})")
            for k,v in res['all_probabilities'].items(): print(f"  {k:8s}: {v:.3f}")
        else:
            print(res)
    elif choice == 2:
        recognizer.evaluate_saved()
    else:
        print("Invalid choice.")
