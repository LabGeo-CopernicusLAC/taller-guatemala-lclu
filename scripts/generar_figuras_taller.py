# -*- coding: utf-8 -*-
"""Figuras reproducibles del Taller Guatemala (sin reprocesar los GeoTIFF).

Ejecutar en el Python de QGIS / OSGeo4W:
    exec(compile(open(str(PROJECT_DIR / 'scripts' / 'generar_figuras_taller.py'), encoding='utf-8').read(),
                 'generar_figuras_taller.py', 'exec'))

También puede ejecutarse desde consola con Python que tenga GDAL, NumPy, Pandas y Matplotlib:
    python scripts/generar_figuras_taller.py
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # no requiere ventana gráfica
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, to_rgba
from matplotlib.patches import Patch
from osgeo import gdal

# Si se ejecuta dentro de qgis_notebook, PROJECT_DIR ya está definido.
# Si se ejecuta como script, la raíz se deduce de scripts/.
try:
    ROOT = Path(PROJECT_DIR).resolve()
except NameError:
    ROOT = Path(__file__).resolve().parent.parent

RASTER = ROOT / 'data' / 'raster'
SALIDAS = RASTER / 'salidas_comparacion'
FIG = ROOT / 'figures'
DIRS = {
    'datos': FIG / '02_datos',
    'cultivos': FIG / '03_visualizacion',
    'comparacion': FIG / '04_comparacion',
    'resultados': FIG / '05_resultados',
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 14, 'axes.titleweight': 'bold',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})

GENERADAS = []


def encontrar_unico(raiz, nombre):
    candidatos = list(raiz.rglob(nombre))
    if len(candidatos) != 1:
        raise FileNotFoundError(f'Se esperaba exactamente un {nombre} en {raiz}: {len(candidatos)} encontrados')
    return candidatos[0]


def leyenda_csv(ruta, codigo, nombre, separador):
    with ruta.open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f, delimiter=separador)
        return {int(float(row[codigo])): ' '.join(row[nombre].split())
                for row in reader if row.get(codigo, '').strip()}


def paleta_codigos(codigos):
    """Colores deterministas por código. No representa la simbología oficial."""
    import colorsys
    return {c: colorsys.hsv_to_rgb(((i * 0.61803398875) % 1.0), 0.72, 0.88)
            for i, c in enumerate(sorted(codigos))}


def resumen_raster(ruta, tamano=1400):
    """Muestreo NEAREST exclusivamente para visualizar, no para cuantificar."""
    ds = gdal.Open(str(ruta), gdal.GA_ReadOnly)
    if ds is None:
        raise RuntimeError(f'No se pudo abrir: {ruta}')
    ancho, alto = ds.RasterXSize, ds.RasterYSize
    escala = min(1.0, tamano / max(ancho, alto))
    nx, ny = max(1, round(ancho * escala)), max(1, round(alto * escala))
    band = ds.GetRasterBand(1)
    array = band.ReadAsArray(buf_xsize=nx, buf_ysize=ny, resample_alg=gdal.GRIORA_NearestNeighbour)
    if array is None:
        raise RuntimeError(f'No se pudo leer el ráster {ruta}')
    nodata = band.GetNoDataValue()
    gt = ds.GetGeoTransform()
    crs = ds.GetProjectionRef()
    ds = None
    valido = np.isfinite(array)
    if nodata is not None:
        valido &= (~np.isnan(array) if np.isnan(nodata) else array != nodata)
    return array, valido, (gt[0], gt[0] + ancho * gt[1], gt[3] + alto * gt[5], gt[3]), crs


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=210, bbox_inches='tight', facecolor='white', pad_inches=0.20)
    plt.close(fig)
    GENERADAS.append(ruta)
    print('[OK]', ruta.relative_to(ROOT))


def mapa(ruta, destino, titulo, etiquetas, colores=None, mascara=False, max_clases=13):
    if not ruta.is_file():
        print('[PENDIENTE]', ruta.relative_to(ROOT))
        return
    a, valido, extent, _ = resumen_raster(ruta)
    codigos, conteos = np.unique(a[valido].astype(np.int64), return_counts=True)
    seleccion = list(codigos[np.argsort(conteos)[::-1]][:max_clases])
    if mascara:
        seleccion = [0, 1]
    if colores is None:
        colores = paleta_codigos(seleccion)
    # En todas las imágenes, los datos inválidos son transparentes.
    rgba = np.ones(a.shape + (4,), dtype=np.float32)
    rgba[:, :, :3] = 1
    rgba[:, :, 3] = 0
    for cod in seleccion:
        mascara_clase = valido & (a == cod)
        if cod == 0 and mascara:
            continue  # otras clases quedan transparentes para destacar cultivos
        rgba[mascara_clase, :3] = to_rgba(colores.get(int(cod), '#888888'))[:3]
        rgba[mascara_clase, 3] = 1
    otros = valido & ~np.isin(a, seleccion)
    if np.any(otros):
        rgba[otros, :3] = (0.75, 0.75, 0.75)
        rgba[otros, 3] = 1
    fig = plt.figure(figsize=(12, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[3.6, 1.7])
    ax, lax = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax.imshow(rgba, origin='upper', extent=extent, interpolation='nearest')
    ax.set_aspect('equal')
    ax.set_axis_off()
    ax.set_title(titulo, loc='left', pad=15)
    patches = []
    for cod in seleccion:
        if mascara and cod == 0:
            continue
        patches.append(Patch(facecolor=colores.get(int(cod), '#888888'),
                             label=f'{int(cod)} · {etiquetas.get(int(cod), "Clase sin nombre")}'))
    if np.any(otros):
        patches.append(Patch(facecolor='#bfbfbf', label='Otras clases (agrupadas)'))
    if mascara:
        patches.append(Patch(facecolor='white', edgecolor='#bbbbbb', label='Otras clases / fondo'))
    lax.axis('off')
    if patches:
        lax.legend(handles=patches, loc='center left', fontsize=8.3, frameon=False,
                   handlelength=1.4, borderaxespad=0, title='Leyenda', title_fontsize=11)
    fig.text(0.02, 0.01, 'Representación categórica. Colores de apoyo visual (no necesariamente oficiales).',
             fontsize=8, color='#666666')
    guardar(fig, destino)


def codigo(texto):
    m = re.match(r'^\s*(\d+)', str(texto))
    if not m:
        raise ValueError(f'Código no identificable: {texto!r}')
    return int(m.group(1))


def etiqueta_corta(texto):
    return re.sub(r'^\s*\d+\s*-\s*(?:\d+\s*-\s*)?', '', str(texto)).strip()


def contar_cruce(path):
    ds = gdal.Open(str(path))
    if ds is None:
        raise RuntimeError(f'No se pudo abrir {path}')
    band = ds.GetRasterBand(1)
    cuentas = np.zeros(4, dtype=np.int64)
    for y in range(0, ds.RasterYSize, 256):
        block = band.ReadAsArray(0, y, ds.RasterXSize, min(256, ds.RasterYSize - y))
        cuentas += np.bincount(block.ravel().astype(np.int64), minlength=256)[:4]
    ds = None
    return cuentas


def figura_acuerdo(cuentas):
    # Filas: local cultivos/otras, columnas: CRCLAC cultivos/otras.
    z = np.array([[cuentas[1], cuentas[3]], [cuentas[2], cuentas[0]]])
    porcentajes = z / z.sum(axis=1, keepdims=True) * 100
    import matplotlib.colors as mcolors
    colores = [['#238b45', '#6baed6'], ['#fdae6b', '#d9d9d9']]
    im = np.array([[mcolors.to_rgb(v) for v in fila] for fila in colores])
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    ax.imshow(im)
    ax.set_xticks([0, 1], ['Cultivos', 'Otras clases'])
    ax.set_yticks([0, 1], ['Cultivos', 'Otras clases'])
    ax.set_xlabel('Clasificación CRCLAC')
    ax.set_ylabel('Clasificación local')
    ax.set_title('Matriz de acuerdo entre productos\n(píxeles válidos, porcentaje por fila)', pad=17)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f'{z[i,j]:,}\n({porcentajes[i,j]:.1f} %)',
                    ha='center', va='center', fontsize=13, color='#182332')
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=4)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.tight_layout(pad=2)
    guardar(fig, DIRS['resultados'] / 'fig_acuerdo_productos.png')


COLORES_CRCLAC = {
    12288: '#377eb8', 16384: '#8da0cb', 20480: '#fc8d62',
    24576: '#e5c494', 28672: '#66c2a5', 36864: '#a6d854',
    4096: '#e41a1c', 40960: '#ffd92f', 45056: '#984ea3'
}
NOMBRES_CRCLAC = {
    12288: 'Pastizales', 16384: 'Cobertura arbórea',
    20480: 'Cobertura arbustiva', 24576: 'Vegetación herbácea inundada',
    28672: 'Manglares', 36864: 'Suelo desnudo',
    4096: 'Superficies artificiales', 40960: 'Nieve y glaciares',
    45056: 'Cuerpos de agua'
}


def figura_barras():
    path_m = SALIDAS / 'matriz_clases_locales_vs_crclac_solo_local_hectareas.csv'
    path_r = SALIDAS / 'resumen_clases_locales_no_mapeadas.csv'
    if not path_m.is_file() or not path_r.is_file():
        print('[PENDIENTE] CSV de matriz / resumen'); return
    matriz = pd.read_csv(path_m, index_col=0, encoding='utf-8-sig')
    matriz.index = [codigo(v) for v in matriz.index]
    matriz.columns = [codigo(v) for v in matriz.columns]
    matriz = matriz.groupby(level=0).sum()
    matriz = matriz.T.groupby(level=0).sum().T
    resumen = pd.read_csv(path_r, encoding='utf-8-sig')
    resumen.columns = resumen.columns.str.strip()
    resumen['codigo'] = resumen['clase_local'].map(codigo)
    top = (resumen.sort_values('solo_local_ha', ascending=False)
           .drop_duplicates('codigo').head(10).copy())
    faltantes = set(top['codigo']) - set(matriz.index)
    if faltantes:
        raise ValueError(f'Códigos faltantes en la matriz: {faltantes}')
    mat = matriz.loc[top.codigo.tolist()].copy()
    mat = mat.loc[:, mat.sum() > 0]
    mat = mat[mat.sum().sort_values(ascending=False).index]
    y = np.arange(len(top))
    fig, ax = plt.subplots(figsize=(14.5, 8.4))
    left = np.zeros(len(top), dtype=float)
    for c in mat.columns:
        v = mat[c].to_numpy(dtype=float)
        ax.barh(y, v, left=left, color=COLORES_CRCLAC.get(c, '#aaa'),
                height=.73, edgecolor='white', linewidth=.35)
        left += v
    lab = [f'{r.codigo} · {etiqueta_corta(r.clase_local)}' for r in top.itertuples()]
    lab = [x if len(x) < 49 else x.split(' · ')[0] + ' · Otras hortalizas' for x in lab]
    ax.set_yticks(y, lab, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(left) * 1.22)
    for i, v in enumerate(left):
        ax.text(v + max(left)*.012, i, f'{v:,.0f} ha'.replace(',', '.'), va='center', fontsize=9)
    ax.set_xlabel('Superficie de discrepancia (ha)')
    ax.set_title('Clases CRCLAC en las áreas agrícolas identificadas solo por el producto local', pad=18)
    ax.grid(axis='x', alpha=.22, linestyle='--'); ax.set_axisbelow(True)
    for s in ('top', 'right', 'left'): ax.spines[s].set_visible(False)
    ax.tick_params(axis='y', length=0)
    handles = [Patch(facecolor=COLORES_CRCLAC.get(c, '#aaa'), label=NOMBRES_CRCLAC.get(c, str(c)))
               for c in mat.columns]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.52, .02), ncol=3, frameon=False, fontsize=9)
    fig.subplots_adjust(left=.34, right=.96, top=.87, bottom=.25)
    guardar(fig, DIRS['resultados'] / 'fig_clases_solo_local.png')



def limites_departamentales():
    """Dibuja los contornos del GeoJSON sin depender de GeoPandas."""
    archivos = list((ROOT / 'data' / 'shapes').rglob('03_Limites_departamentales.geojson'))
    if len(archivos) != 1:
        print('[PENDIENTE] límites departamentales: GeoJSON no encontrado'); return
    with archivos[0].open(encoding='utf-8') as f:
        geo = json.load(f)
    fig, ax = plt.subplots(figsize=(9, 8))
    n = 0
    for feat in geo.get('features', []):
        geom = feat.get('geometry') or {}
        typ, coords = geom.get('type'), geom.get('coordinates', [])
        polygons = [coords] if typ == 'Polygon' else (coords if typ == 'MultiPolygon' else [])
        for poly in polygons:
            for ring in poly:
                pts = np.asarray(ring)
                if pts.ndim == 2 and pts.shape[1] >= 2:
                    ax.plot(pts[:, 0], pts[:, 1], color='#24618d', lw=.7)
        n += 1
    ax.set_aspect('equal')
    ax.set_xlabel('Longitud')
    ax.set_ylabel('Latitud')
    ax.set_title(f'Guatemala · límites departamentales ({n} entidades)')
    ax.legend(handles=[Patch(facecolor='none', edgecolor='#24618d', label='Límites departamentales')],
              loc='lower left', frameon=False)
    fig.tight_layout()
    guardar(fig, DIRS['datos'] / 'limites_departamentales.png')

def referencias_rmd():
    """Fragmento listo para pegar en un .Rmd ubicado en rmarkdown/."""
    salida = ROOT / 'rmarkdown' / 'referencias_figuras_generadas.Rmd'
    salida.parent.mkdir(parents=True, exist_ok=True)
    lines = ['# Figuras del taller', '', 'Figuras producidas a partir de los rásteres y CSV del procesamiento.', '']
    for i, ruta in enumerate(GENERADAS, 1):
        relativa = '../' + ruta.relative_to(ROOT).as_posix()
        lines.extend([f'## Figura {i}. {ruta.stem.replace("_", " ").capitalize()}', '',
                      f'```{{r fig_taller_{i:02d}, echo=FALSE, out.width="95%", fig.align="center"}}',
                      f'knitr::include_graphics("{relativa}")', '```', ''])
    salida.write_text('\n'.join(lines), encoding='utf-8')
    print('[OK]', salida.relative_to(ROOT))


def main():
    crclac_leyenda = leyenda_csv(RASTER/'crclac-2020'/'legend.csv', 'Value', 'Name (ES)', ',')
    local_path = encontrar_unico(RASTER, 'landcover_guatemala_2020_20m.tif')
    local_leyenda = leyenda_csv(local_path.parent/'legend.csv', 'C_USO', 'USOT', ';')
    limites_departamentales()
    tareas = [
        (RASTER/'crclac-2020'/'mosaic.vrt', DIRS['datos']/'crclac_mosaico_original.png',
         'CRCLAC 2020 · mosaico original', crclac_leyenda, None, False),
        (SALIDAS/'crclac_2020_area_estudio.tif', DIRS['datos']/'crclac_recortado_mapa.png',
         'CRCLAC 2020 · recorte de Guatemala', crclac_leyenda, None, False),
        (local_path, DIRS['datos']/'landcover_local_2020_mapa.png',
         'Land Cover Guatemala 2020 · clasificación local', local_leyenda, None, False),
        (SALIDAS/'crclac_cultivos.tif', DIRS['cultivos']/'cultivos_crclac.png',
         'Cultivos en CRCLAC 2020', {1:'Cultivos'}, {1:'#31a354'}, True),
        (SALIDAS/'local_cultivos.tif', DIRS['cultivos']/'cultivos_local.png',
         'Cultivos en Land Cover Guatemala 2020', {1:'Cultivos'}, {1:'#31a354'}, True),
        (SALIDAS/'crclac_clases_alineadas_a_local.tif', DIRS['comparacion']/'crclac_alineado_mapa.png',
         'CRCLAC 2020 · clases alineadas a grilla local (20 m)', crclac_leyenda, None, False),
        (SALIDAS/'cruce_cultivos_crclac_vs_local.tif', DIRS['comparacion']/'cruce_cultivos.png',
         'Cruce agrícola · CRCLAC vs Land Cover Guatemala 2020',
         {0:'Ninguno identifica cultivos',1:'Ambos identifican cultivos',
          2:'Solo CRCLAC identifica cultivos',3:'Solo local identifica cultivos'},
         {0:'#d9d9d9',1:'#238b45',2:'#fdae6b',3:'#6baed6'}, False),
    ]
    for args in tareas:
        mapa(*args)
    cross = SALIDAS/'cruce_cultivos_crclac_vs_local.tif'
    if cross.is_file():
        figura_acuerdo(contar_cruce(cross))
    else:
        print('[PENDIENTE] matriz de acuerdo: falta cruce raster')
    figura_barras()
    referencias_rmd()
    print(f'\nFiguras generadas: {len(GENERADAS)}')


if __name__ == '__main__' or 'PROJECT_DIR' in globals():
    main()
