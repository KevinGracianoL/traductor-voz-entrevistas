"""Tests de setup_dlls — corren siempre (tmp_path, sin venv real)."""

import runpy
import sys
import time
import types
from pathlib import Path

import pytest

from setup_dlls import dirs_cuda, escribir_keeper, instalar, plan_dlls


class _Handle:
    """Handle falso observable (os.add_dll_directory no existe en Linux)."""

    def __init__(self, path: str) -> None:
        self.path = path


class _FakeOs(types.ModuleType):
    """os real salvo add_dll_directory, que registra y devuelve handle observable."""

    def __init__(self, real_os: types.ModuleType) -> None:
        super().__init__("os")
        self.__dict__["_real_os"] = real_os
        self.registrados: list[str] = []

    def __getattr__(self, name: str) -> object:
        return getattr(self.__dict__["_real_os"], name)

    def add_dll_directory(self, path: str) -> _Handle:
        self.registrados.append(path)
        return _Handle(path)


def _fake_os() -> _FakeOs:
    import os as real_os

    return _FakeOs(real_os)


def test_dirs_cuda_ordena_destino_primero(tmp_path: Path) -> None:
    import os

    bins = dirs_cuda("/nvidia", str(tmp_path))
    assert bins[0] == str(tmp_path)
    esperados = [os.path.join("/nvidia", sub, "bin") for sub in ("cublas", "cudnn", "cuda_runtime")]
    assert bins[1:] == esperados


def test_plan_dlls_destino_y_solo_3_a_ct2(tmp_path: Path) -> None:
    archivos = {"cublas": ["a.dll", "x.txt"], "cudnn": ["b.dll"], "cuda_runtime": ["c.dll"]}
    plan = plan_dlls("/nvidia", str(tmp_path), str(tmp_path / "ct2"), archivos)
    a_dest = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path))
    assert a_dest == ["a.dll", "b.dll", "c.dll"]
    a_ct2 = sorted(Path(src).name for src, dst in plan if dst == str(tmp_path / "ct2"))
    assert a_ct2 == ["cublas64_12.dll", "cublasLt64_12.dll", "cudart64_12.dll"]


def test_escribir_keeper_contenido(tmp_path: Path) -> None:
    keeper, pth = escribir_keeper(str(tmp_path), ["D1", "D2"])
    texto = Path(keeper).read_text(encoding="utf-8")
    compile(texto, keeper, "exec")
    assert "_HANDLES = [os.add_dll_directory(p) for p in _DIRS]" in texto
    assert "'D1'" in texto and "'D2'" in texto
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"
    assert Path(keeper).parent == tmp_path
    assert Path(pth).parent == tmp_path


def test_keeper_ejecuta_y_conserva_handles(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El keeper generado registra cada dir y conserva los handles (también en Linux)."""
    sub = tmp_path / "bin"
    sub.mkdir()
    keeper, _ = escribir_keeper(str(tmp_path), [str(sub)])
    fake_os = _fake_os()
    monkeypatch.setitem(sys.modules, "os", fake_os)
    ns = runpy.run_path(str(keeper))
    assert fake_os.registrados == [str(sub)]
    assert len(ns["_HANDLES"]) == 1
    assert ns["_HANDLES"][0].path == str(sub)


def test_instalar_copia_y_registra_en_arbol_falso(tmp_path: Path) -> None:
    """instalar() end-to-end contra tmp: copias reales + keeper ejecutable."""
    nvidia = tmp_path / "nvidia"
    for sub, files in {
        "cublas": ["a.dll", "cublas64_12.dll", "cublasLt64_12.dll", "x.txt"],
        "cudnn": ["b.dll"],
        "cuda_runtime": ["c.dll", "cudart64_12.dll"],
    }.items():
        bindir = nvidia / sub / "bin"
        bindir.mkdir(parents=True)
        for f in files:
            (bindir / f).write_bytes(b"fake-dll")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    site_packages = tmp_path / "site-packages"
    site_packages.mkdir()
    ct2 = site_packages / "ctranslate2"
    ct2.mkdir()
    keeper, pth = instalar(str(nvidia), str(dest), str(site_packages), str(ct2))
    assert (dest / "a.dll").exists()
    assert not (dest / "x.txt").exists()
    assert (ct2 / "cublas64_12.dll").exists()
    assert not (ct2 / "b.dll").exists()
    assert Path(pth).read_text(encoding="utf-8").strip() == "import zz_nvidia_dlls"
    assert Path(keeper).exists()
    assert Path(keeper).parent == site_packages
    assert Path(pth).parent == site_packages
    assert repr(str(dest)) in Path(keeper).read_text(encoding="utf-8")


def test_copiar_dlls_reemplaza_existente(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Re-ejecución: el viejo se renombra a .tmp, se copia el nuevo y se limpia."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"viejo")
    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(a: str, b: str) -> None:
        renames.append((a, b))
        real_replace(a, b)

    monkeypatch.setattr(os, "replace", spy_replace)
    copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"
    assert (str(dest / "cublas64_12.dll"), str(dest / "cublas64_12.dll.tmp")) in renames
    assert not (dest / "cublas64_12.dll.tmp").exists()
    assert (ct2 / "cublas64_12.dll").exists()


def test_copiar_dlls_sin_destino_previo_no_renombra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    renames: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(a: str, b: str) -> None:
        renames.append((a, b))
        real_replace(a, b)

    monkeypatch.setattr(os, "replace", spy_replace)
    copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert renames == []
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"


def test_copiar_dlls_reintenta_y_acierta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Lock transitorio (AV): el rename falla 2 veces y luego pasa, con backoff."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"bloqueado")
    real_replace = os.replace
    intentos: list[str] = []
    duerme: list[float] = []

    def replace_que_se_desbloquea(a: str, b: str) -> None:
        intentos.append(a)
        if len(intentos) <= 2:
            raise PermissionError(a)
        real_replace(a, b)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(os, "replace", replace_que_se_desbloquea)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=5, espera=0)
    assert (dest / "cublas64_12.dll").read_bytes() == b"origen"
    assert len(intentos) == 3
    assert duerme == [0.0, 0.0]


def test_copiar_dlls_agota_reintentos_y_lanza(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"bloqueado")
    intentos: list[str] = []

    def bloqueado(a: str, b: str) -> None:
        intentos.append(a)
        raise PermissionError(a)

    monkeypatch.setattr(os, "replace", bloqueado)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=3, espera=0)
    assert len(intentos) == 3


def test_copiar_dlls_reintentos_por_defecto(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Contrato por defecto: 6 intentos con backoff de 1.0 s."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"bloqueado")
    intentos: list[str] = []
    duerme: list[float] = []

    def bloqueado(a: str, b: str) -> None:
        intentos.append(a)
        raise PermissionError(a)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(os, "replace", bloqueado)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    with pytest.raises(RuntimeError, match="no puedo reemplazar"):
        copiar_dlls(str(nvidia), str(dest), str(ct2))
    assert len(intentos) == 6
    assert duerme == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_copiar_dlls_limpia_tmp_con_reintento(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """El .tmp viejo (todavía escaneado) se borra con reintento, no se abandona."""
    import os

    from setup_dlls import copiar_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"origen")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"origen")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"origen")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    ct2 = tmp_path / "ct2"
    ct2.mkdir()
    (dest / "cublas64_12.dll").write_bytes(b"viejo")
    real_remove = os.remove
    duerme: list[float] = []
    intentos: list[str] = []

    def remove_que_se_desbloquea(path: str) -> None:
        intentos.append(path)
        if len(intentos) <= 2:
            raise PermissionError(path)
        real_remove(path)

    def spy_sleep(s: float) -> None:
        duerme.append(s)

    monkeypatch.setattr(os, "remove", remove_que_se_desbloquea)
    monkeypatch.setattr(time, "sleep", spy_sleep)
    copiar_dlls(str(nvidia), str(dest), str(ct2), reintentos=5, espera=0)
    assert not (dest / "cublas64_12.dll.tmp").exists()
    assert len(intentos) == 3
    assert duerme == [0.0, 0.0]


def test_dir_ct2_deriva_del_paquete_real(monkeypatch: pytest.MonkeyPatch) -> None:
    """Sin paquete instalado falla claro (ModuleNotFoundError), no fantasma."""
    import sys
    import types

    from setup_dlls import dir_ct2

    fake = types.ModuleType("ctranslate2")
    fake.__file__ = "/fake/site/ctranslate2/__init__.py"
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    assert dir_ct2() == "/fake/site/ctranslate2"


def test_dir_ct2_sin_file_lanza(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import types

    from setup_dlls import dir_ct2

    fake = types.ModuleType("ctranslate2")
    fake.__file__ = None
    monkeypatch.setitem(sys.modules, "ctranslate2", fake)
    with pytest.raises(RuntimeError) as excinfo:
        dir_ct2()
    assert str(excinfo.value) == "ctranslate2 sin __file__: instalación rota"


def test_main_directo_con_overrides(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """main() directo con overrides: atribuible por mutmut, cero toques reales."""
    import sys
    import types

    from setup_dlls import main

    monkeypatch.setitem(sys.modules, "nvidia", types.ModuleType("nvidia"))

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    dest = tmp_path / "Scripts"
    dest.mkdir()
    sp = tmp_path / "sp"
    sp.mkdir()
    ct2 = sp / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    main(str(nvidia), str(dest), str(sp))
    assert (dest / "cublas64_12.dll").exists()
    assert (ct2 / "cudart64_12.dll").exists()
    out = capsys.readouterr().out
    assert out == f"keeper+pth en {sp / 'zz_nvidia_dlls.py'} y {sp / 'zz_nvidia_dlls.pth'}\n"


def test_main_defaults_usan_fakes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() sin args resuelve todo desde nvidia/exe fakes (rama defaults)."""
    import sys
    import types

    from setup_dlls import main

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(nvidia)]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    exe = tmp_path / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(sys, "executable", str(exe))
    ct2 = tmp_path / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    main()
    assert (tmp_path / "Scripts" / "cublas64_12.dll").exists()
    assert (ct2 / "cudart64_12.dll").exists()


def test_guard_main_solo_bajo_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """El guard __main__ dispara main() con fakes; importar no ejecuta nada."""
    import runpy
    import sys
    import types

    import setup_dlls

    nvidia = tmp_path / "nvidia"
    (nvidia / "cublas" / "bin").mkdir(parents=True)
    (nvidia / "cublas" / "bin" / "cublas64_12.dll").write_bytes(b"f")
    (nvidia / "cublas" / "bin" / "cublasLt64_12.dll").write_bytes(b"f")
    (nvidia / "cudnn" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin").mkdir(parents=True)
    (nvidia / "cuda_runtime" / "bin" / "cudart64_12.dll").write_bytes(b"f")
    fake_nvidia = types.ModuleType("nvidia")
    fake_nvidia.__path__ = [str(nvidia)]
    monkeypatch.setitem(sys.modules, "nvidia", fake_nvidia)
    exe = tmp_path / "Scripts" / "python.exe"
    exe.parent.mkdir(parents=True)
    exe.touch()
    monkeypatch.setattr(sys, "executable", str(exe))
    ct2 = tmp_path / "ctranslate2"
    ct2.mkdir()
    fake_ct2 = types.ModuleType("ctranslate2")
    fake_ct2.__file__ = str(ct2 / "__init__.py")
    monkeypatch.setitem(sys.modules, "ctranslate2", fake_ct2)
    raiz = Path(setup_dlls.__file__)
    runpy.run_path(str(raiz), run_name="__main__")
    assert (tmp_path / "Scripts" / "cublas64_12.dll").exists()


def test_keeper_escapa_rutas_raras(tmp_path: Path) -> None:
    """Apóstrofos y no-ASCII no rompen el keeper generado (va en UTF-8)."""
    rara = tmp_path / "O'Brien-é"
    rara.mkdir()
    keeper, _ = escribir_keeper(str(tmp_path), [str(rara)])
    raw = Path(keeper).read_bytes()
    texto = raw.decode("utf-8")
    compile(texto, keeper, "exec")
    assert repr(str(rara)) in texto
    assert "é".encode() in raw
