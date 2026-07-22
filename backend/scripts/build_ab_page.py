"""Build a self-contained A/B listening page from the generated excerpts.

Reads the OGG excerpts, measures real identity-similarity + brightness for each,
embeds the audio as data URIs, and writes a dark 'steering console' HTML page.
Run in the base env (needs librosa + remixflow.analysis).
"""
import base64
import html
import json

import numpy as np

from remixflow.audio import analysis
from remixflow.audio.io import load

SCR = "/tmp/claude-1000/-data3-remixflow/eebcdae0-d882-407a-89c2-f62d2eb7fb47/scratchpad"
OUT_HTML = f"{SCR}/remixflow_ab.html"

VERSIONS = [
    ("original", "Original", "The source excerpt (0:45–1:15).", 0, ""),
    ("subtle", "Subtle", "Nearly identical — melody & rhythm locked.", 20,
     "faithful cover, same style and mood"),
    ("moderate", "Moderate", "Rockier, brighter, more energy.", 50,
     "rock, energetic, bright"),
    ("bold", "Bold", "Reimagined — jazz, warm, intricate.", 80,
     "jazz, warm, intricate"),
]


def b64_mp3(name):
    with open(f"{SCR}/ab_{name}.mp3", "rb") as f:
        return base64.b64encode(f.read()).decode()


import librosa

def mel(name):
    y, sr = librosa.load(f"{SCR}/ab_{name}.mp3", sr=22050, mono=True)
    return librosa.power_to_db(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64))

# "Change from original" = 1 - mel-spectrogram correlation (a perceptual measure
# of how different it actually sounds, unlike the saturating MFCC identity score).
orig_mel = mel("original")

def change_pct(name):
    if name == "original":
        return 0
    v = mel(name)
    n = min(orig_mel.shape[1], v.shape[1])
    corr = float(np.corrcoef(orig_mel[:, :n].flatten(), v[:, :n].flatten())[0, 1])
    return max(0, round((1.0 - corr) * 100))

cards = []
for name, title, desc, variation, prompt in VERSIONS:
    cards.append({
        "name": name, "title": title, "desc": desc, "variation": variation,
        "prompt": prompt, "change": change_pct(name), "audio": b64_mp3(name),
    })

data_json = json.dumps([
    {k: c[k] for k in ("name", "title", "variation", "change")} for c in cards
])

# --- markup ---------------------------------------------------------------
chips = "\n".join(
    f'<button class="chip" data-i="{i}" role="tab" aria-selected="{str(i==0).lower()}">'
    f'<span class="chip-name">{html.escape(c["title"])}</span>'
    f'<span class="chip-var">{c["variation"]}%</span></button>'
    for i, c in enumerate(cards)
)

rows = "\n".join(
    f'''<article class="card" data-i="{i}" tabindex="0" aria-label="Select {html.escape(c['title'])}">
      <div class="card-head">
        <div>
          <h3>{html.escape(c['title'])}</h3>
          <p class="desc">{html.escape(c['desc'])}</p>
        </div>
        <div class="badge">{'source' if c['name']=='original' else 'ACE-Step v1.5'}</div>
      </div>
      {('<p class="prompt"><span>prompt</span>' + html.escape(c['prompt']) + '</p>') if c['prompt'] else '<p class="prompt src">unmodified reference</p>'}
      <div class="meters">
        <div class="meter">
          <div class="m-head"><span>Variation</span><span class="tnum">{c['variation']}%</span></div>
          <div class="track"><div class="fill violet" style="width:{c['variation']}%"></div></div>
        </div>
        <div class="meter">
          <div class="m-head"><span>Change from original</span><span class="tnum">{c['change']}%</span></div>
          <div class="track"><div class="fill teal" style="width:{c['change']}%"></div></div>
        </div>
      </div>
    </article>'''
    for i, c in enumerate(cards)
)

audio_tags = "\n".join(
    f'<audio id="a{i}" preload="auto" src="data:audio/mpeg;base64,{c["audio"]}"></audio>'
    for i, c in enumerate(cards)
)

HTML = f"""<title>RemixFlow — Listen · I Will Never Fall</title>
<style>
:root {{
  --bg:#0a0c14; --bg2:#0e1120; --panel:#151a28; --panel2:#1c2233; --line:#28304a;
  --ink:#eef1f8; --muted:#8b93ab; --accent:#7c9cff; --violet:#a78bfa; --teal:#2dd4bf;
  --shadow: 0 18px 50px -20px rgba(0,0,0,.7);
}}
* {{ box-sizing:border-box; }}
html,body {{ margin:0; }}
body {{
  background:
    radial-gradient(1100px 520px at 82% -8%, rgba(124,156,255,.14), transparent),
    radial-gradient(820px 480px at -5% 108%, rgba(167,139,250,.13), transparent),
    var(--bg);
  color:var(--ink); min-height:100vh;
  font-family:ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  -webkit-font-smoothing:antialiased; line-height:1.5;
}}
.tnum {{ font-variant-numeric:tabular-nums; }}
.wrap {{ max-width:840px; margin:0 auto; padding:40px 24px 72px; }}

header .eyebrow {{ text-transform:uppercase; letter-spacing:.22em; font-size:12px; color:var(--muted); margin:0 0 10px; }}
header h1 {{ font-size:clamp(30px,5vw,46px); line-height:1.04; margin:0 0 8px; font-weight:760; letter-spacing:-.02em; text-wrap:balance; }}
header h1 .glow {{ background:linear-gradient(92deg,var(--teal),var(--accent) 55%,var(--violet)); -webkit-background-clip:text; background-clip:text; color:transparent; }}
header .sub {{ color:var(--muted); font-size:15px; margin:0; max-width:60ch; }}
.legend {{ display:flex; gap:18px; flex-wrap:wrap; margin:18px 0 30px; font-size:13px; color:var(--muted); }}
.legend b {{ color:var(--ink); font-weight:600; }}
.dot {{ display:inline-block; width:9px; height:9px; border-radius:50%; vertical-align:middle; margin-right:6px; }}
.dot.violet {{ background:var(--violet); }} .dot.teal {{ background:var(--teal); }}

/* transport */
.transport {{ background:linear-gradient(180deg,var(--panel),var(--bg2)); border:1px solid var(--line); border-radius:18px; padding:20px; box-shadow:var(--shadow); position:sticky; top:16px; z-index:5; backdrop-filter:blur(6px); }}
.np {{ display:flex; align-items:center; gap:16px; }}
.play {{ flex:none; width:56px; height:56px; border-radius:50%; border:none; cursor:pointer; background:linear-gradient(135deg,var(--accent),var(--violet)); color:#0a0c14; display:grid; place-items:center; transition:transform .12s, filter .12s; }}
.play:hover {{ filter:brightness(1.08); transform:scale(1.04); }}
.play svg {{ width:22px; height:22px; }}
.np-info {{ flex:1; min-width:0; }}
.np-title {{ font-weight:650; font-size:17px; }}
.np-meta {{ color:var(--muted); font-size:13px; }}
.time {{ font-size:12px; color:var(--muted); font-variant-numeric:tabular-nums; flex:none; }}
.seek {{ margin-top:14px; height:8px; border-radius:99px; background:var(--panel2); cursor:pointer; position:relative; overflow:hidden; }}
.seek-fill {{ position:absolute; inset:0 auto 0 0; width:0; background:linear-gradient(90deg,var(--accent),var(--violet)); border-radius:99px; }}
.chips {{ display:flex; gap:8px; margin-top:16px; flex-wrap:wrap; }}
.chip {{ display:flex; align-items:center; gap:8px; background:var(--panel2); border:1px solid var(--line); color:var(--ink); border-radius:99px; padding:8px 14px; cursor:pointer; font-size:13px; transition:.14s; }}
.chip:hover {{ border-color:var(--accent); }}
.chip[aria-selected="true"] {{ background:var(--accent); border-color:var(--accent); color:#0a0c14; font-weight:640; }}
.chip-var {{ font-variant-numeric:tabular-nums; opacity:.7; font-size:12px; }}
.chip[aria-selected="true"] .chip-var {{ opacity:.85; }}
.hint {{ font-size:12.5px; color:var(--muted); margin:12px 2px 0; }}

/* cards */
.cards {{ display:flex; flex-direction:column; gap:14px; margin-top:28px; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:16px; padding:18px 20px; transition:.16s; cursor:pointer; }}
.card:hover {{ border-color:#37415f; }}
.card.active {{ border-color:var(--accent); box-shadow:inset 3px 0 0 var(--accent), var(--shadow); }}
.card:focus-visible {{ outline:2px solid var(--accent); outline-offset:2px; }}
.card-head {{ display:flex; justify-content:space-between; align-items:flex-start; gap:12px; }}
.card h3 {{ margin:0; font-size:18px; font-weight:680; letter-spacing:-.01em; }}
.desc {{ margin:2px 0 0; color:var(--muted); font-size:13.5px; }}
.badge {{ flex:none; font-size:11px; text-transform:uppercase; letter-spacing:.08em; color:var(--muted); border:1px solid var(--line); border-radius:99px; padding:4px 10px; }}
.prompt {{ font-family:ui-monospace,"SF Mono",Menlo,monospace; font-size:12.5px; color:#b9c2dc; background:var(--bg2); border:1px solid var(--line); border-radius:9px; padding:8px 11px; margin:14px 0; }}
.prompt span {{ text-transform:uppercase; letter-spacing:.1em; color:var(--muted); font-size:10px; margin-right:9px; }}
.prompt.src {{ color:var(--muted); font-style:normal; }}
.meters {{ display:grid; grid-template-columns:1fr 1fr; gap:16px; }}
.m-head {{ display:flex; justify-content:space-between; font-size:12px; color:var(--muted); margin-bottom:6px; }}
.track {{ height:7px; background:var(--panel2); border-radius:99px; overflow:hidden; }}
.fill {{ height:100%; border-radius:99px; }}
.fill.violet {{ background:linear-gradient(90deg,#7c3aed,var(--violet)); }}
.fill.teal {{ background:linear-gradient(90deg,#0f9e8e,var(--teal)); }}

footer {{ margin-top:34px; color:var(--muted); font-size:12.5px; text-align:center; line-height:1.7; }}
footer code {{ font-family:ui-monospace,monospace; color:#b9c2dc; }}
@media (max-width:520px) {{ .meters {{ grid-template-columns:1fr; }} }}
@media (prefers-reduced-motion:reduce) {{ * {{ transition:none !important; }} }}
</style>

<div class="wrap">
  <header>
    <p class="eyebrow">RemixFlow · Listen</p>
    <h1>Keep everything you love.<br><span class="glow">Just make it slightly different.</span></h1>
    <p class="sub">One 30-second excerpt of <strong>“I Will Never Fall,”</strong> regenerated by ACE-Step v1.5 at three levels of variation. Same tempo, same key — the higher the variation, the further it drifts from the source while staying recognizably itself.</p>
    <div class="legend">
      <span><span class="dot violet"></span><b>Variation</b> — the setting we asked for</span>
      <span><span class="dot teal"></span><b>Change from original</b> — how different it measurably sounds</span>
    </div>
  </header>

  <section class="transport" aria-label="Player">
    <div class="np">
      <button class="play" id="play" aria-label="Play">
        <svg id="ic-play" viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>
      </button>
      <div class="np-info">
        <div class="np-title" id="np-title">Original</div>
        <div class="np-meta" id="np-meta">Variation 0% · Change 0%</div>
      </div>
      <div class="time"><span id="cur">0:00</span> / <span id="dur">0:30</span></div>
    </div>
    <div class="seek" id="seek"><div class="seek-fill" id="seek-fill"></div></div>
    <div class="chips" role="tablist" aria-label="Versions">
      {chips}
    </div>
    <p class="hint">Switch versions while it plays — the playhead stays put, so you're always comparing the same moment.</p>
  </section>

  <div class="cards">
    {rows}
  </div>

  <footer>
    Generated locally by RemixFlow on 2× RTX 4060&nbsp;Ti · ACE-Step v1.5 (SDEdit img2img)<br>
    Steering → <code>variation_amount</code> = noise level · sliders → text prompt · identity locks → stronger preservation
  </footer>
</div>

{audio_tags}

<script>
const DATA = {data_json};
const audios = DATA.map((_, i) => document.getElementById('a'+i));
let cur = 0, playing = false;
const $ = id => document.getElementById(id);
const fmt = s => (isFinite(s) ? Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0') : '0:00');

function select(i, keepPlaying) {{
  const t = audios[cur].currentTime;
  audios[cur].pause();
  cur = i;
  const a = audios[cur];
  if (isFinite(t)) a.currentTime = Math.min(t, a.duration || t);
  document.querySelectorAll('.chip').forEach((c,j)=>c.setAttribute('aria-selected', j===i));
  document.querySelectorAll('.card').forEach((c,j)=>c.classList.toggle('active', j===i));
  $('np-title').textContent = DATA[i].title;
  $('np-meta').textContent = 'Variation '+DATA[i].variation+'% · Change '+DATA[i].change+'%';
  if (keepPlaying && playing) a.play();
}}
function setPlaying(p) {{
  playing = p;
  $('ic-play').innerHTML = p
    ? '<path d="M6 5h4v14H6zM14 5h4v14h-4z"/>'
    : '<path d="M8 5v14l11-7z"/>';
  $('play').setAttribute('aria-label', p ? 'Pause' : 'Play');
}}
$('play').addEventListener('click', () => {{
  if (playing) {{ audios[cur].pause(); setPlaying(false); }}
  else {{ audios[cur].play(); setPlaying(true); }}
}});
document.querySelectorAll('.chip').forEach((c,i)=>c.addEventListener('click',()=>select(i,true)));
document.querySelectorAll('.card').forEach((c,i)=>{{
  c.addEventListener('click',()=>select(i,true));
  c.addEventListener('keydown',e=>{{ if(e.key==='Enter'||e.key===' '){{e.preventDefault();select(i,true);}} }});
}});
audios.forEach(a=>{{
  a.addEventListener('timeupdate',()=>{{
    if (a!==audios[cur]) return;
    const d=a.duration||30;
    $('seek-fill').style.width=(100*a.currentTime/d)+'%';
    $('cur').textContent=fmt(a.currentTime); $('dur').textContent=fmt(d);
  }});
  a.addEventListener('ended',()=>{{ if(a===audios[cur]) setPlaying(false); }});
}});
$('seek').addEventListener('click', e => {{
  const r=e.currentTarget.getBoundingClientRect();
  const a=audios[cur]; a.currentTime=((e.clientX-r.left)/r.width)*(a.duration||30);
}});
select(0,false);
</script>
"""

with open(OUT_HTML, "w") as f:
    f.write(HTML)
print("wrote", OUT_HTML)
print("change%:", [(c["title"], c["variation"], c["change"]) for c in cards])
