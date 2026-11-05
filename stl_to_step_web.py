"""STL -> STEP 변환 웹페이지 (로컬 서버, 표준 라이브러리만).

실행:  python stl_to_step_web.py [--port 8765] [--host 127.0.0.1]
열기:  http://127.0.0.1:8765
같은 와이파이의 폰에서 쓰려면 --host 0.0.0.0 으로 실행하고 PC IP 로 연다.
"""
import argparse
import json
import os
import sys
import tempfile
import threading
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stl_to_step import convert  # noqa: E402

MAX_BYTES = 200 * 1024 * 1024
LOCK = threading.Lock()  # OCC 는 한 번에 하나씩

PAGE = r"""<!DOCTYPE html>
<html lang="ko"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>STL → STEP 변환기</title>
<style>
 body{font-family:system-ui,"Malgun Gothic",sans-serif;max-width:760px;margin:40px auto;padding:0 16px;color:#222;background:#f6f7f9}
 h1{font-size:24px;margin-bottom:4px} p.sub{color:#666;margin-top:0}
 #drop{border:2px dashed #8aa;border-radius:12px;padding:44px;text-align:center;background:#fff;cursor:pointer;transition:.15s}
 #drop.over{background:#e8f3ff;border-color:#2b7de9}
 .opts{margin:14px 0;display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-size:14px}
 .opts input[type=number]{width:70px}
 table{width:100%;border-collapse:collapse;background:#fff;border-radius:8px;overflow:hidden;font-size:14px;margin-top:14px}
 th,td{padding:8px 10px;border-bottom:1px solid #eee;text-align:left}
 th{background:#eef1f5} .ok{color:#18794e} .err{color:#c62828} .warn{color:#b26a00}
 a.btn{background:#2b7de9;color:#fff;padding:4px 10px;border-radius:6px;text-decoration:none}
 .note{font-size:13px;color:#666;margin-top:18px;line-height:1.6}
</style></head><body>
<h1>STL → STEP 변환기</h1>
<p class="sub">STL 파일을 끌어다 놓으면 STEP 으로 바꿔 줍니다 (내 PC 에서 처리, 밖으로 안 나감).</p>
<div id="drop">여기에 STL 파일을 끌어다 놓거나 <b>클릭해서 고르기</b> (여러 개 가능)
 <input id="file" type="file" accept=".stl" multiple hidden></div>
<div class="opts">
 <label><input id="merge" type="checkbox" checked> 같은 평면 면 합치기</label>
 <label>꿰매기 오차 <input id="tol" type="number" value="0.01" step="0.001" min="0.0001"> mm</label>
</div>
<table id="tbl" hidden><thead><tr><th>파일</th><th>상태</th><th>결과</th><th></th></tr></thead><tbody></tbody></table>
<div class="note">
 · 곡면은 매끈한 곡면이 아니라 삼각형 조각 그대로 들어갑니다.<br>
 · 삼각형 20만 개가 넘는 큰 파일은 수십 분 걸릴 수 있습니다 (먼저 메시를 줄이기 권장).<br>
 · 파일은 한 번에 하나씩 차례로 처리됩니다.
</div>
<script>
const drop=document.getElementById('drop'),inp=document.getElementById('file'),tb=document.querySelector('#tbl tbody');
drop.onclick=()=>inp.click();
inp.onchange=()=>run([...inp.files]);
['dragenter','dragover'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.add('over')}));
['dragleave','drop'].forEach(e=>drop.addEventListener(e,ev=>{ev.preventDefault();drop.classList.remove('over')}));
drop.addEventListener('drop',ev=>run([...ev.dataTransfer.files]));
async function run(files){
  document.getElementById('tbl').hidden=false;
  for(const f of files){
    const tr=document.createElement('tr');
    tr.innerHTML=`<td>${f.name}</td><td>대기</td><td>${(f.size/1048576).toFixed(1)} MB</td><td></td>`;
    tb.appendChild(tr);
    f._tr=tr;
  }
  for(const f of files){
    const c=f._tr.children;
    if(!/\.stl$/i.test(f.name)){c[1].innerHTML='<span class=err>STL 아님</span>';continue}
    c[1].textContent='변환 중…'; const t0=Date.now();
    const timer=setInterval(()=>c[1].textContent=`변환 중… ${((Date.now()-t0)/1000).toFixed(0)} s`,1000);
    try{
      const q=`?name=${encodeURIComponent(f.name)}&tol=${document.getElementById('tol').value}&merge=${document.getElementById('merge').checked?1:0}`;
      const r=await fetch('/convert'+q,{method:'POST',body:f});
      const j=await r.json(); clearInterval(timer);
      if(!r.ok) throw new Error(j.error||r.status);
      const w=[]; if(j.open) w.push(`열린 껍질 ${j.open}개`); if(!j.valid) w.push('형상 검사 불통과');
      c[1].innerHTML=w.length?`<span class=warn>완료 (경고: ${w.join(', ')})</span>`:'<span class=ok>완료</span>';
      c[2].textContent=`삼각형 ${j.tri.toLocaleString()} → 면 ${j.faces.toLocaleString()} · 덩어리 ${j.bodies} · ${j.vol.toFixed(1)} cm³ · ${j.sec.toFixed(1)} s`;
      c[3].innerHTML=`<a class=btn href="/download/${j.id}" download="${j.out}">STEP 받기</a>`;
    }catch(e){clearInterval(timer);c[1].innerHTML=`<span class=err>실패: ${e.message}</span>`}
  }
}
</script></body></html>"""

OUTDIR = tempfile.mkdtemp(prefix="stl2step_")
RESULTS = {}  # id -> (path, download name)


class H(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            return self._send(200, PAGE, "text/html; charset=utf-8")
        if self.path.startswith("/download/"):
            rid = self.path.split("/")[-1]
            if rid in RESULTS:
                path, name = RESULTS[rid]
                with open(path, "rb") as fh:
                    data = fh.read()
                cd = "attachment; filename*=UTF-8''" + urllib.parse.quote(name)
                return self._send(200, data, "application/step", {"Content-Disposition": cd})
        self._send(404, {"error": "없음"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        if u.path != "/convert":
            return self._send(404, {"error": "없음"})
        q = urllib.parse.parse_qs(u.query)
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0 or n > MAX_BYTES:
            return self._send(400, {"error": f"파일 크기 문제 ({n} B, 최대 200 MB)"})
        name = os.path.basename(q.get("name", ["part.stl"])[0])
        tol = float(q.get("tol", ["0.01"])[0])
        merge = q.get("merge", ["1"])[0] == "1"
        rid = os.urandom(8).hex()
        src = os.path.join(OUTDIR, rid + ".stl")
        dst = os.path.join(OUTDIR, rid + ".step")
        with open(src, "wb") as fh:
            left = n
            while left:
                chunk = self.rfile.read(min(left, 1 << 20))
                if not chunk:
                    break
                fh.write(chunk)
                left -= len(chunk)
        try:
            with LOCK:
                r = convert(src, dst, tol, merge)
        except Exception as e:
            return self._send(500, {"error": str(e)})
        finally:
            os.remove(src)
        out = os.path.splitext(name)[0] + ".step"
        RESULTS[rid] = (dst, out)
        r.update(id=rid, out=out)
        print(f"변환: {name} -> 면 {r['faces']} ({r['sec']:.1f}s)", flush=True)
        self._send(200, r)

    def log_message(self, *a):
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()
    print(f"STL->STEP 변환기: http://{a.host}:{a.port}  (임시 폴더 {OUTDIR})", flush=True)
    ThreadingHTTPServer((a.host, a.port), H).serve_forever()


if __name__ == "__main__":
    main()
