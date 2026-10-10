# Experimento: CaP-X en Hanoi con RGB y propiocepción

La fase 1 valida el dataset y fija el protocolo. La fase 2 implementa el
adaptador OCBench y comprueba la ejecución de movimientos mediante CaP-X. La
fase 3 implementa la evaluación del agente restaurando estados finales de las
demostraciones. El entrenamiento de un modelo sigue fuera de este experimento.

## Dataset y fuentes:

Dataset: [NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2](https://huggingface.co/datasets/NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2).
Revisión fijada: `c5f40db6f68a71147bef8224273dfeeb9d06bd81`.
Contiene 225 episodios, 933889 registros y tres vídeos RGB de 224 x 224 por
instante, distribuidos en once archivos MP4. La exportación es LeRobot v3.

El generador revisado es
`/home/luigidu/ocbench_leonardo/ocbench/scripts/generate_hanoi_lerobot.py`,
en el commit `95e3d4e16756f864b58e32a6a661286288ed3be8`. Su SHA-256 y las
versiones del entorno están en [protocol.json](protocol.json).
Los 89 archivos Python y XML de `ocbench/` coinciden con el checkout
`/home/luigidu/murobotics_ws/ocbench`, commit
`e2cd2f72110b66bd65afab1b855d81ebc73aeacc`.

La documentación específica del exportador está en
`/home/luigidu/ocbench_leonardo/ocbench/scripts/README.md`.
`/home/luigidu/murobotics_ws/ocbench/experiments/README.md` explica las
variantes CPU, lite y visual, pero su experimento BC/FBC corresponde a Block.

## Contrato de observaciones y acciones:

El modelo recibe `front`, `side`, `wrist` y siete posiciones del robot:
seis articulaciones en radianes y la apertura normalizada de la pinza.
La convención de pinza es 0 cerrada y 1 abierta.
No recibe poses privilegiadas de objetos o postes, estado del oracle,
recompensas, flags de éxito ni observaciones futuras.

El exportador lee estado y RGB antes de ejecutar la acción. La acción guardada
es el objetivo del actuador, no la posición medida después del paso. Los
campos `next.reward`, `next.done` y `next.success` describen el resultado de
ese paso.

Para un objetivo `target` y la posición medida actual `current`, la conversión
de vuelta al entorno es:

```python
scale = np.array([0.18, 0.18, 0.18, 0.36, 0.36, 0.36, 0.12])
env_action = np.clip((target - current) / scale, -1.0, 1.0)
observation, reward, terminated, truncated, info = env.step(env_action)
```

Los límites absolutos del brazo deben venir de `actuator_ctrlrange`, nunca
de los mínimos y máximos de las demostraciones.

## Variante experimental:

El usuario eligió RGB + propiocepción. Esta combinación no es el protocolo
oficial de observaciones exclusivamente visuales de OCBench.

| Propiedad | Protocolo fijado |
| --- | --- |
| Task | `hanoi-triple-task2`, sin lite |
| Backend inicial | CPU MuJoCo |
| Entorno del generador | `hanoi-cpu-triple-task2-v0` |
| Referencia visual registrada | `visual-hanoi-cpu-triple-task2-v0` |
| Control y vídeos | 30 Hz |
| Integración física | 20 subpasos, 1/600 s por subpaso |
| Horizonte | 7000 acciones, 233.333333 s simulados |
| Inicio | Tres discos en el poste central, layout aleatorio |
| Objetivo | Torre legal completa en cualquiera de los dos postes exteriores |
| Veredicto | Éxito oficial, saludable y sin fallo |

El generador cambia los valores predeterminados de OCBench: control a 50 Hz
e integración física a 0.002 s. Los resultados de este protocolo deben
identificarse como variante adaptada al dataset, sin presentarlos como una
reproducción directa de las tablas oficiales. El tiempo de inferencia del LLM
no es tiempo físico simulado; debe medirse por separado.

## Evidencia de validación:

[validation_summary.json](validation_summary.json) resume el alcance y las
comprobaciones realizadas, incluido el comando completo del auditor.

- [numeric_validation.json](numeric_validation.json): todos los registros,
  índices, límites de incremento, aperturas, timestamps y finales de episodio
  pasaron. Hay exactamente un `done` y un `success` final en cada episodio.
- [actuator_validation.json](actuator_validation.json): 103 casos comparan el
  objetivo del exportador con los comandos reales de MuJoCo. Error máximo
  `1.1707160707530306e-07`.
- [episode_provenance.json](episode_provenance.json): 225 semillas candidatas
  reconstruidas, con coincidencia exacta de propiocepción inicial. Hay 225
  layouts reconstruidos diferentes.
- [replay.json](replay.json): reproducción completa de los episodios
  0, 56, 112, 168 y 224. Los cinco terminan con éxito, saludables, sin fallo
  ni truncación y con la misma longitud que sus demostraciones.
- [video_probe.json](video_probe.json): cabeceras y primer frame de los once
  MP4. Cada cámara declara en total 933889 frames, H.264 y 30 fps.
- [initial_rgb_alignment.json](initial_rgb_alignment.json): nueve frames de
  los episodios 0, 112 y 224 comparados con renders de las escenas
  reconstruidas. Los timestamps coinciden; el PSNR es 39.29 a 43.67 dB,
  compatible con las diferencias de compresión H.264 observadas.

El generador no persiste la semilla aceptada ni el orden de jobs. Con varios
workers, el orden LeRobot corresponde al orden de terminación y no al índice
original del job. La reconstrucción usa la hipótesis de semilla raíz 0 y los
reintentos del generador; las coincidencias de estado, los nueve frames y las
cinco reproducciones son la evidencia disponible. No sustituye un manifiesto
original de generación. Las futuras recolecciones deben guardar los argumentos,
commit, versiones, job, semilla aceptada, intento y estado inicial completos.

El 225/225 de éxitos del dataset describe demostraciones seleccionadas.
No es una medida del rendimiento en episodios nuevos. No se reprodujeron
completamente los 225 episodios ni se decodificaron todos los frames de video.

## Divisiones y evaluación:

[split_manifest.json](split_manifest.json) fija 180 episodios de entrenamiento,
22 de validación y 23 de prueba, con semilla de partición `20261009`.
Se separan episodios completos y sus layouts reconstruidos. Todas las ventanas
o programas derivados de un episodio deben conservar su división.

[evaluation_scenes.json](evaluation_scenes.json) fija además 50 escenas nuevas
de validación y 100 de prueba en simulación, con semillas y hashes del estado
inicial. Sus semillas y layouts no coinciden con los reconstruidos del dataset
ni entre los dos conjuntos nuevos. Los 23 episodios offline de prueba y las
100 escenas nuevas del simulador tienen funciones distintas.

El futuro adaptador debe usar exactamente la semilla de reset del manifiesto
y comprobar su hash inicial. La semilla derivada por el framework para un
trial se registra aparte; no debe sustituir silenciosamente ese reset fijado.

La prueba final ejecutará el modelo base y el ajustado sobre las mismas 100
escenas, sin filtrar fallos ni reintentar hasta conseguir éxito. La métrica
principal será éxitos divididos entre las 100 escenas programadas; los errores
de ejecución de la policy cuentan como intentos fallidos y se informan aparte.
Las métricas internas condicionadas a trials sin error de `inspect-robots`
no deben reemplazar ese denominador. `FINISH` no constituye un veredicto de
éxito.

## Repetir la auditoría:

Descargar los metadatos y Parquet de la revisión fijada. Por ejemplo, con
el entorno de OCBench que contiene `huggingface_hub`:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2",
    repo_type="dataset",
    revision="c5f40db6f68a71147bef8224273dfeeb9d06bd81",
    allow_patterns=["meta/**", "data/**"],
    local_dir="/tmp/capx-hanoi-snapshot",
)
```

Desde la raíz de Inspect Robots, usando el entorno del generador:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /home/luigidu/ocbench_leonardo/ocbench/.venv/bin/python -B \
  experiments/capx_hanoi/validate.py \
  --dataset-root /tmp/capx-hanoi-snapshot \
  --ocbench-root /home/luigidu/ocbench_leonardo/ocbench \
  --output-dir /tmp/capx-hanoi-revalidation \
  --probe-videos
```

La entrada permanece intacta. El sondeo de vídeos consulta MP4 remotos
de la revisión fijada; necesita red y PyAV. Los datos numéricos y la
simulación se validan localmente. La auditoría inicial usó el entorno con
las versiones guardadas en `protocol.json`.

## Siguiente fase:

Implementar el adaptador OCBench y comprobar el control a través de CaP-X.
La interpolación de `MotionQueue`, la cadencia y la apertura continua de la
pinza necesitan pruebas propias. Esta fase no incluye etiquetas Python,
entrenamiento, despliegue en Modal ni resultados de un modelo aprendido.

## Fase 2: adaptador y ejecución CaP-X:

El plugin [inspect-robots-ocbench](../../plugins/inspect-robots-ocbench/README.md)
registra el embodiment `ocbench` y la task `ocbench-hanoi`. Implementa el
protocolo anterior sin modificar el core. Comprueba los límites UR5e, la escala
de deltas y el hash del mundo inicial. La semilla del manifiesto controla el
reset; la semilla derivada por el framework queda registrada por separado.
La geometría privilegiada del manifiesto no se copia al `Scene` entregado a
la política.

CaP-X tiene integrado el cliente del LLM y su conversación con el ejecutor,
pero no incluye un modelo ni pesos entrenables. Solo requiere un LLM externo
capaz de interpretar imágenes y producir Python. El ciclo es:

```text
RGB + joint_pos + instrucción
  -> LLM configurado: genera Python
  -> CaP-X: ejecuta Python y construye ActionChunk
  -> guardrails -> adaptador OCBench -> simulación
  -> nuevas observaciones + feedback de ejecución -> LLM
```

`camera="front"` selecciona la cámara para SAM3. El mensaje visual del LLM
incluye las tres cámaras disponibles. Los servidores SAM3, Contact-GraspNet y
Pyroki no son otros LLM supervisores. Solo se llaman cuando el Python utiliza
sus helpers. Este protocolo no entrega profundidad ni extrínsecos calibrados,
por lo que `plan_grasp` no dispone de sus entradas necesarias. La validación
de movimientos usa objetivos articulares y no llama a esos servidores.

Se añadió `move_to_configuration([j0, j1, j2, j3, j4, j5, opening])` a CaP-X.
Permite mover simultáneamente el brazo y la pinza, conservando aperturas
intermedias y los límites de interpolación. `move_to_joints`, `open_gripper` y
`close_gripper` siguen disponibles.

[validate_movements.py](validate_movements.py) devuelve código pregrabado por
`httpx.MockTransport` y lo procesa con `CapxPolicy` real. Ejecuta los helpers,
los guardrails de bounds y delta, la conversión de acciones y la física de
OCBench. Los bloques predeterminados contienen 64 objetivos y se renderizan
las tres cámaras antes de cada nueva llamada. Dentro de cada bloque se reutiliza
RGB, mientras el estado y la física se actualizan en cada paso. Este ahorro de
renderizado se limita al validador. La evaluación de un modelo real utiliza
el adaptador normal.

[phase2_summary.json](phase2_summary.json) resume la validación. Los cinco
episodios probados completaron Hanoi, con 16940 pasos y 265 llamadas al
modelo simulado. Ningún guardrail modificó una acción. La interpolación
añadió 100 pasos respecto a las demostraciones originales.

| Episodio | Pasos originales | Pasos CaP-X | Llamadas simuladas | Resultado |
| --- | --- | --- | --- | --- |
| 0 | 3058 | 3079 | 48 | Éxito |
| 56 | 4408 | 4437 | 69 | Éxito |
| 112 | 2856 | 2870 | 45 | Éxito |
| 168 | 3273 | 3289 | 52 | Éxito |
| 224 | 3245 | 3265 | 51 | Éxito |

Los resultados individuales están en
[movement_validation_chunk64.json](movement_validation_chunk64.json).
[movement_validation.json](movement_validation.json) registra adicionalmente
el episodio 0 como un solo bloque de 3058 objetivos, también exitoso.

La reproducción no mide percepción, razonamiento ni generalización del LLM.
Los objetivos proceden de episodios conocidos. Su éxito demuestra que el
camino de ejecución admite completar la task. Las escenas nuevas del
manifiesto siguen reservadas para evaluar el modelo.

## Preparar el entorno de simulación:

El entorno `/tmp/capx-hanoi-env` quedó instalado y probado: se verificó el
reset de `validation-000` y se repitió el episodio 0 con éxito, registrado en
[isolated_runtime_validation.json](isolated_runtime_validation.json). Las
versiones usadas están en `phase2_summary.json`. `/tmp` es temporal; los
comandos siguientes permiten recrearlo.

Conviene usar un entorno separado del workspace para mantener las versiones
de física y NumPy del protocolo. Ejecutar desde la raíz de `inspect-robots`:

```bash
uv venv --python 3.12 /tmp/capx-hanoi-env
uv pip install --python /tmp/capx-hanoi-env/bin/python \
  numpy==2.2.6 mujoco==3.15.0 dm-control==1.0.48 gymnasium==1.4.0 \
  -e /home/luigidu/ocbench_leonardo/ocbench \
  -e . -e plugins/inspect-robots-agent \
  -e plugins/inspect-robots-capx -e plugins/inspect-robots-ocbench
```

Antes de instalar, verificar que el checkout OCBench corresponde al commit
fijado en `protocol.json`. Para desarrollo del plugin sin simulación:
`uv sync --locked --all-packages --extra dev`.

El archivo numérico de la auditoría anterior está en
`/tmp/capx-hanoi-audit/numeric.npz`; no está incluido en Git y `/tmp` no es
almacenamiento permanente. Puede regenerarse con el auditor de la fase 1.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /tmp/capx-hanoi-env/bin/python experiments/capx_hanoi/validate_movements.py \
  --numeric /tmp/capx-hanoi-audit/numeric.npz \
  --episodes 0 56 112 168 224 --chunk-size 64 \
  --output /tmp/capx-hanoi-movement-check.json
```

## Evaluar el modelo configurado:

[run.py](run.py) evalúa escenas nuevas con guardrails, guarda imágenes y
transcripciones, y escribe un resumen junto a los logs. La métrica principal
es éxitos divididos entre todas las escenas programadas, incluyendo errores.
`evaluation_status="success"` solo indica que la evaluación terminó sin error;
la resolución de Hanoi se mide con `success_rate_all_scheduled`.

El endpoint puede estar desplegado en Modal o ser una API de proveedor. Usar
el identificador exacto que acepte ese endpoint. Los nombres de modelos que
aparecen en Codex no establecen sus identificadores ni disponibilidad en una
API pública. La clave se configura en el entorno; `--api-key-env` recibe el
nombre de la variable, nunca la clave.

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /tmp/capx-hanoi-env/bin/python experiments/capx_hanoi/run.py \
  --model "$CAPX_MODEL" --base-url "$CAPX_BASE_URL" \
  --api-key-env CAPX_API_KEY --wire chat \
  --split validation --num-scenes 1 --log-dir logs/capx-hanoi-baseline
```

El modelo debe aceptar imágenes y generar código. Cambiar `--wire` a
`responses` o `anthropic` cuando corresponda al endpoint configurado. Primero
usar validación para ajustar la política; reservar `--split test` para comparar
el modelo base y el ajustado con las mismas 100 escenas.

## Fase 3: reset agentico desde episodios resueltos:

[run_reset.py](run_reset.py) ejecuta el protocolo de reset sin una VLA. El
backend se elige explícitamente con `--policy`: `agent` es el agente nativo y
predeterminado; `capx` conserva el baseline opcional de código generado. Para
cada escena seleccionada de `split_manifest.json`:

```text
reset OCBench con la semilla reconstruida
  -> capturar front, side y wrist como referencia privada del objetivo
  -> reproducir todas las acciones del episodio hasta su Hanoi resuelto
  -> reiniciar únicamente el contador de pasos del agente
  -> entregar referencia + RGB actual + joint_pos a la política seleccionada
  -> evaluar después de cada acción la torre en el poste inicial
```

La reproducción prepara el estado inicial de la evaluación. Sus acciones no
llegan al LLM y sus pasos no se mezclan con `episode_length`. OCBench se crea
con `terminate_at_success=False`, por lo que alcanzar el objetivo de la tarea
original no reinicia el entorno. El criterio de reset usa el poste inicial del
layout aleatorio, exige los tres discos en orden, la pinza liberada y cinco
pasos de control consecutivos en esa condición. No exige que el brazo copie su
pose de la imagen de referencia.

El hash bitwise del mundo inicial se conserva en la provenance, junto con
`initial_world_hash_matches`. En el reset es diagnóstico y no bloqueante:
componentes internos float64 de MuJoCo e IK pueden variar entre runtimes aunque
la propiocepción exportada y el layout coincidan. El gate bloqueante es más
directo para este protocolo: reproducir todas las acciones debe mantenerse
saludable y terminar en éxito oficial de OCBench. La task normal de Hanoi
conserva su comprobación estricta del hash.

[reset_setup_smoke.json](reset_setup_smoke.json) registra una comprobación con
OCBench real del primer episodio de validación: el episodio 2 reprodujo sus
4303 acciones, terminó resuelto y saludable, dejó el objetivo de reset aún
pendiente y produjo las tres referencias visuales. Esta prueba no hizo ninguna
petición a un LLM.

La instrucción enviada al modelo es positiva y especifica el resultado:

> Restore the three disks to the original middle-peg stack shown in the
> reference images using an efficient sequence of safe moves. Place the largest
> disk on the bottom, the medium disk in the middle, and the smallest disk on
> top. Release the disks and leave the stack stable. The robot arm may finish
> in any safe pose.

Esto sigue el patrón de tareas de manipulación de describir directamente el
estado objetivo. El veredicto no depende del texto ni de `FINISH`; lo calcula
el simulador con estado privilegiado que nunca se incorpora a la observación.

Tanto `agent` como `capx` envían las imágenes de referencia sólo en la primera
llamada del trial, claramente separadas de las cámaras actuales. Después
permanecen en el historial del modelo. Para `agent`, el conjunto nativo de
herramientas se deriva del action space: en esta variante joint-space son
`move_joints`, `done` y `give_up` (más `take_pic` si se selecciona
`--images on_demand`). CaP-X no se importa ni necesita servidores cuando se
usa `--policy agent`. Los logs guardan frames, acciones, transcripción, éxito y
longitud del intento.

### Preparar los datos:

Sólo hacen falta `meta/**` y `data/**`, no los vídeos. Mantener la revisión
fijada:

```bash
/tmp/capx-hanoi-env/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="NONHUMAN-RESEARCH/ocbench-hanoi-triple-task2",
    repo_type="dataset",
    revision="c5f40db6f68a71147bef8224273dfeeb9d06bd81",
    allow_patterns=["meta/**", "data/**"],
    local_dir="/tmp/capx-hanoi-snapshot",
)
PY
```

El entorno de ejecución necesita `pyarrow` para seleccionar las acciones de los
episodios. Si no está instalado:

```bash
uv pip install --python /tmp/capx-hanoi-env/bin/python pyarrow huggingface-hub
```

### Configurar la API y ejecutar el agente nativo:

La clave se coloca en una variable de entorno. No se escribe en archivos del
repositorio ni se pasa literalmente por CLI:

```bash
export OPENAI_API_KEY='tu-clave-de-proyecto'
```

Prueba mínima con un episodio de validación:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /tmp/capx-hanoi-env/bin/python experiments/capx_hanoi/run_reset.py \
  --dataset-root /tmp/capx-hanoi-snapshot \
  --policy agent --model gpt-6-astra --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY --wire responses --effort low \
  --max-llm-calls 8 \
  --split validation --num-scenes 1 \
  --log-dir logs/ocbench-hanoi-reset-agent-astra-8calls
```

El modelo debe admitir imágenes y function calling. GPT-6 Astra requiere
`--wire responses` para las herramientas. Con `--images always` (el valor por
defecto) recibe las tres cámaras actuales en cada turno; no se levanta ningún
servidor CaP-X.

Para comparar después con CaP-X, seleccionar el backend explícitamente:

```bash
env -u PYTHONPATH PYTHONNOUSERSITE=1 MUJOCO_GL=egl \
  /tmp/capx-hanoi-env/bin/python experiments/capx_hanoi/run_reset.py \
  --dataset-root /tmp/capx-hanoi-snapshot \
  --policy capx --model gpt-6-astra --base-url https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY --wire responses \
  --max-llm-calls 8 \
  --split validation --num-scenes 1 \
  --log-dir logs/capx-hanoi-reset-astra-smoke
```

La clave se crea y administra en la configuración del proyecto de OpenAI. El
runner sólo lee el nombre indicado por `--api-key-env`; nunca persiste su valor.
La [documentación oficial de autenticación](https://platform.openai.com/docs/api-reference/introduction)
recomienda cargar claves desde variables de entorno, y la
[ficha oficial de GPT-6 Astra](https://developers.openai.com/api/docs/models/gpt-6-astra)
confirma el identificador `gpt-6-astra`, entrada visual y soporte de Responses.

Las métricas principales son `success_rate_all_scheduled` y
`mean_agent_steps_on_success`. Los errores permanecen en el denominador. El
replay de preparación queda registrado como `setup_replay_steps` dentro de la
provenance, separado de los pasos del agente.

### Alcance del baseline:

Este flujo queda listo para probar razonamiento visual y control joint-space.
OCBench no proporciona depth ni extrínsecos en este protocolo, así que
`plan_grasp` no tiene sus entradas. El modelo puede usar las imágenes, el estado
articular y `move_to_configuration`, pero el éxito autónomo todavía debe medirse.
Una primitive de grasp privilegiada cambiaría la pregunta experimental y no se
incluye silenciosamente en este baseline.

## Trabajo posterior para el objetivo entrenado:

El reset ya tiene runner y evaluación. Todavía falta elegir un modelo y endpoint
concretos, medir el baseline, construir ejemplos de entrenamiento
`observación + instrucción -> Python` con los splits fijados, entrenar y
comparar ambas políticas en las escenas nuevas. El LeRobot actual contiene
acciones articulares; no contiene etiquetas Python ni conversaciones CaP-X.
Un bloque de objetivos se puede expresar con el nuevo helper, pero elegir el
horizonte y verificar su reproducción forma parte de construir esas etiquetas.

Modal puede alojar el entrenamiento y el servidor de un modelo cuyos pesos
podamos ajustar. Configurar una API externa en CaP-X permite evaluar ese modelo;
no da acceso a sus pesos para entrenarlos en Modal.

El workflow incluye el nuevo job de CI y el job de publicación del plugin.
Antes de publicar por primera vez habrá que configurar el entorno GitHub
`pypi-ocbench` y su trusted publisher en PyPI. No se publicó ningún paquete.
