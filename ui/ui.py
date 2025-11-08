"""Aplicación gráfica para visualizar canales de archivos EDF.

Esta interfaz permite cargar un archivo EDF, seleccionar qué canales mostrar
y graficarlos directamente en la ventana utilizando Matplotlib incrustado en
Tkinter. También permite ajustar de forma interactiva la escala temporal y de
amplitud para acercar (zoom) la visualización, y resaltar intervalos de
convulsiones cargados desde archivos CSV de anotaciones.
"""

from __future__ import annotations

import csv
import json
import logging
import threading
from pathlib import Path
from typing import Any

try:
	import tkinter as tk
	from tkinter import filedialog, messagebox
except ImportError as tk_import_error:  # pragma: no cover - entornos sin Tk
	tk = None
	filedialog = None
	messagebox = None
	TK_IMPORT_ERROR = tk_import_error
else:
	TK_IMPORT_ERROR = None

import numpy as np
from matplotlib.figure import Figure

if tk is not None:
	from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
else:  # pragma: no cover - entornos sin Tk
	FigureCanvasTkAgg = None  # type: ignore[assignment]
	NavigationToolbar2Tk = None  # type: ignore[assignment]

try:
	import pyedflib
except ImportError as import_error:  # pragma: no cover - solo para contexto sin dependencia
	EDF_IMPORT_ERROR = import_error
	pyedflib = None
else:
	EDF_IMPORT_ERROR = None

try:
	import requests
except ImportError:
	requests = None


LOGGER = logging.getLogger(__name__)

# Fuentes y tamaños por defecto para una UI más legible
DEFAULT_FONT = ("TkDefaultFont", 11)
BOLD_FONT = ("TkDefaultFont", 12, "bold")
SMALL_FONT = ("TkDefaultFont", 10)

SEIZURE_LABEL = "seiz"
PREDICTED_LABEL = "predicted_seizure"
API_BASE_URL = "http://127.0.0.1:8000/api/models"


def load_edf_file(file_path: str | Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
	"""Carga un archivo EDF y devuelve un diccionario con sus señales.

	Parameters
	----------
	file_path: str | Path
		Ruta absoluta o relativa al archivo EDF.

	Returns
	-------
	dict[str, tuple[np.ndarray, np.ndarray]]
		Diccionario donde cada clave es el nombre de un canal y el valor una
		tupla *(tiempo, señal)*.

	Raises
	------
	ImportError
		Si la dependencia ``pyEDFlib`` no está instalada en el entorno.
	FileNotFoundError
		Si la ruta proporcionada no existe.
	ValueError
		Si el archivo no contiene canales válidos.
	"""

	if pyedflib is None:
		raise ImportError(
			"pyEDFlib no está disponible. Instala las dependencias indicadas en requirements.txt"
		) from EDF_IMPORT_ERROR

	path = Path(file_path).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"No se encontró el archivo EDF: {path}")

	LOGGER.info("Cargando archivo EDF: %s", path)
	channel_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}

	reader = pyedflib.EdfReader(str(path))
	try:
		labels = reader.getSignalLabels()
		if not labels:
			raise ValueError("El archivo EDF no contiene canales disponibles")

		for index, label in enumerate(labels):
			samples = reader.readSignal(index)
			sample_rate = float(reader.getSampleFrequency(index)) or 1.0
			time_axis = np.arange(samples.shape[0]) / sample_rate
			channel_data[label] = (time_axis, samples)

	finally:
		reader.close()

	return channel_data


def load_annotation_csv(file_path: str | Path) -> list[tuple[float, float, str, str]]:
	"""Carga un archivo CSV con anotaciones de convulsiones.

	Se esperan archivos en el formato ``csv_v1.0.0`` descrito en el ejemplo, con
	comentarios que comienzan con ``#`` y un encabezado estándar
	``channel,start_time,stop_time,label,confidence``. Se devuelven únicamente los
	intervalos cuya etiqueta ``label`` es ``seiz`` (sin diferenciar mayúsculas/minúsculas).

	Parameters
	----------
	file_path: str | Path
		Ruta absoluta o relativa al archivo CSV de anotaciones.

	Returns
	-------
	list[tuple[float, float, str, str]]
		Lista de tuplas ``(inicio, fin, canal, etiqueta)`` para cada intervalo ``seiz``.

	Raises
	------
	FileNotFoundError
		Si la ruta proporcionada no existe.
	ValueError
		Si no se encuentran columnas de tiempo o no hay intervalos válidos.
	"""

	path = Path(file_path).expanduser().resolve()
	if not path.exists():
		raise FileNotFoundError(f"No se encontró el archivo de anotaciones: {path}")

	intervals: list[tuple[float, float, str, str]] = []
	headers: list[str] | None = None

	with path.open("r", encoding="utf-8", newline="") as csv_file:
		reader = csv.reader(csv_file)
		for row in reader:
			if not row:
				continue
			if row[0].startswith("#"):
				continue

			if headers is None:
				headers = [value.strip().lower() for value in row if value is not None]
				continue

			row_values = {headers[index]: (row[index].strip() if index < len(row) else "") for index in range(len(headers))}
			label = row_values.get("label", "").lower()
			if label != SEIZURE_LABEL:
				continue

			try:
				start = float(row_values.get("start_time", ""))
				stop = float(row_values.get("stop_time", ""))
			except ValueError as exc:
				LOGGER.warning("Fila de anotación inválida en %s: %s", path.name, row)
				continue

			if stop <= start:
				LOGGER.debug("Intervalo descartado por duración no positiva: start=%s stop=%s", start, stop)
				continue

			channel = row_values.get("channel", "")
			intervals.append((start, stop, channel, label))

	if headers is None or "start_time" not in headers or "stop_time" not in headers:
		raise ValueError("El archivo de anotaciones no contiene las columnas requeridas 'start_time' y 'stop_time'.")

	LOGGER.info("Cargadas %d anotaciones 'seiz' desde %s", len(intervals), path.name)
	return intervals


class EDFViewerApp:
	"""Aplicación principal para la visualización de canales EDF."""

	def __init__(self, master: Any) -> None:
		self.master = master
		self.master.title("Visor EDF")
		# Tamaño de ventana aumentado para mejorar la usabilidad
		self.master.geometry("1400x900")

		self.channel_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
		self.loaded_file: Path | None = None
		self.time_max: float = 0.0
		self.annotations: list[tuple[float, float, str, str]] = []
		self.annotation_file: Path | None = None
		self.annotations_enabled = tk.BooleanVar(master=self.master, value=True)
		self.annotation_toggle: Any | None = None
		self.prediction_events: list[tuple[float, float, str, float | None]] = []
		self.prediction_source: str | None = None
		self.predictions_enabled = tk.BooleanVar(master=self.master, value=True)
		self.prediction_toggle: Any | None = None
		self.prediction_var = tk.StringVar(master=self.master, value="Sin predicciones")
		self.metrics_var = tk.StringVar(master=self.master, value="")

		self._build_layout()

	# ------------------------------------------------------------------
	# Construcción de la interfaz
	# ------------------------------------------------------------------
	def _build_layout(self) -> None:
		self.master.columnconfigure(0, weight=0)
		self.master.columnconfigure(1, weight=1)
		self.master.rowconfigure(0, weight=1)

		sidebar = tk.Frame(self.master, padx=10, pady=10)
		sidebar.grid(row=0, column=0, sticky="ns")
		sidebar.rowconfigure(4, weight=1)

		tk.Label(sidebar, text="Archivo EDF:", font=BOLD_FONT)\
			.grid(row=0, column=0, sticky="w")

		self.file_label = tk.Label(sidebar, text="Ningún archivo cargado", wraplength=320, font=DEFAULT_FONT)
		self.file_label.grid(row=1, column=0, pady=(0, 10), sticky="w")

		load_button = tk.Button(sidebar, text="Cargar archivo…", command=self.request_file, font=DEFAULT_FONT)
		load_button.grid(row=2, column=0, sticky="ew", pady=(0, 10))

		tk.Label(sidebar, text="Canales disponibles:", font=BOLD_FONT)\
			.grid(row=3, column=0, sticky="sw")

		self.channel_listbox = tk.Listbox(
			sidebar,
			selectmode=tk.MULTIPLE,
			exportselection=False,
			width=45,
			height=24,
			font=DEFAULT_FONT,
		)
		self.channel_listbox.grid(row=4, column=0, sticky="nsew")

		scrollbar = tk.Scrollbar(sidebar, orient=tk.VERTICAL, command=self.channel_listbox.yview)
		scrollbar.grid(row=4, column=1, sticky="ns")
		self.channel_listbox.configure(yscrollcommand=scrollbar.set)

		plot_button = tk.Button(sidebar, text="Graficar canales seleccionados", command=self.plot_selected, font=DEFAULT_FONT)
		plot_button.grid(row=5, column=0, sticky="ew", pady=(10, 0))

		downsample_label = tk.Label(sidebar, text="Factor de downsampling (>=1):", font=DEFAULT_FONT)
		downsample_label.grid(row=6, column=0, sticky="w", pady=(15, 0))

		self.downsample_var = tk.IntVar(value=1)
		downsample_entry = tk.Spinbox(sidebar, from_=1, to=100, textvariable=self.downsample_var, width=6, font=DEFAULT_FONT)
		downsample_entry.grid(row=7, column=0, sticky="w")

		# Controles de escala temporal (zoom horizontal)
		start_label = tk.Label(sidebar, text="Inicio (s):", font=DEFAULT_FONT)
		start_label.grid(row=8, column=0, sticky="w", pady=(15, 0))
		self.time_start_var = tk.DoubleVar(value=0.0)
		self.time_start_spin = tk.Spinbox(
			sidebar,
			from_=0.0,
			to=0.0,
			increment=0.5,
			textvariable=self.time_start_var,
			width=8,
			font=DEFAULT_FONT,
		)
		self.time_start_spin.grid(row=9, column=0, sticky="w")

		window_label = tk.Label(sidebar, text="Ventana (s):", font=DEFAULT_FONT)
		window_label.grid(row=10, column=0, sticky="w", pady=(10, 0))
		self.time_window_var = tk.DoubleVar(value=0.0)
		self.time_window_spin = tk.Spinbox(
			sidebar,
			from_=0.5,
			to=0.5,
			increment=0.5,
			textvariable=self.time_window_var,
			width=8,
			font=DEFAULT_FONT,
		)
		self.time_window_spin.grid(row=11, column=0, sticky="w")

		# Control de escala vertical (zoom vertical)
		y_label = tk.Label(sidebar, text="Escala Y (x):", font=DEFAULT_FONT)
		y_label.grid(row=12, column=0, sticky="w", pady=(10, 0))
		self.y_scale_var = tk.DoubleVar(value=1.0)
		self.y_scale_spin = tk.Spinbox(
			sidebar,
			from_=0.1,
			to=50.0,
			increment=0.1,
			textvariable=self.y_scale_var,
			width=8,
			font=DEFAULT_FONT,
		)
		self.y_scale_spin.grid(row=13, column=0, sticky="w")

		apply_button = tk.Button(sidebar, text="Actualizar escala", command=self._try_refresh_plot, font=DEFAULT_FONT)
		apply_button.grid(row=14, column=0, sticky="ew", pady=(10, 0))

		tk_annot = tk.Label(sidebar, text="Anotaciones (CSV):", font=BOLD_FONT)\
			.grid(row=15, column=0, sticky="w", pady=(20, 0))
		self.annotation_var = tk.StringVar(master=self.master, value="Ningún CSV cargado")
		self.annotation_label = tk.Label(sidebar, textvariable=self.annotation_var, wraplength=320, justify="left", font=DEFAULT_FONT)
		self.annotation_label.grid(row=16, column=0, sticky="w", pady=(2, 5))

		annotation_button = tk.Button(sidebar, text="Cargar anotaciones…", command=self.request_annotation_file, font=DEFAULT_FONT)
		annotation_button.grid(row=17, column=0, sticky="ew")

		clear_button = tk.Button(sidebar, text="Limpiar anotaciones", command=self.clear_annotations, font=DEFAULT_FONT)
		clear_button.grid(row=18, column=0, sticky="ew", pady=(5, 0))

		self.annotation_toggle = tk.Checkbutton(
			sidebar,
			text="Mostrar anotaciones",
			variable=self.annotations_enabled,
			command=self._on_annotation_toggle,
			font=DEFAULT_FONT,
		)
		self.annotation_toggle.grid(row=19, column=0, sticky="w", pady=(5, 0))
		self.annotation_toggle.configure(state=tk.DISABLED)

		tk.Label(sidebar, text="Predicciones del modelo:", font=BOLD_FONT)\
			.grid(row=20, column=0, sticky="w", pady=(20, 0))
		self.prediction_label = tk.Label(sidebar, textvariable=self.prediction_var, wraplength=320, justify="left", font=DEFAULT_FONT)
		self.prediction_label.grid(row=21, column=0, sticky="w", pady=(2, 5))

		self.prediction_toggle = tk.Checkbutton(
			sidebar,
			text="Mostrar predicciones",
			variable=self.predictions_enabled,
			command=self._on_prediction_toggle,
			font=DEFAULT_FONT,
		)
		self.prediction_toggle.grid(row=22, column=0, sticky="w")
		self.prediction_toggle.configure(state=tk.DISABLED)

		button_frame = tk.Frame(sidebar)
		button_frame.grid(row=23, column=0, sticky="ew", pady=(10, 0))
		button_frame.columnconfigure(0, weight=1)
		button_frame.columnconfigure(1, weight=1)

		self.eval_pt_button = tk.Button(button_frame, text="Evaluar PyTorch", command=lambda: self.evaluate_model("pt"), font=DEFAULT_FONT)
		self.eval_pt_button.grid(row=0, column=0, sticky="ew", padx=(0, 5))
		self.eval_tf_button = tk.Button(button_frame, text="Evaluar TensorFlow", command=lambda: self.evaluate_model("tf"), font=DEFAULT_FONT)
		self.eval_tf_button.grid(row=0, column=1, sticky="ew")

		m_label = tk.Label(sidebar, text="Métricas del modelo:", font=BOLD_FONT)\
			.grid(row=24, column=0, sticky="w", pady=(15, 0))
		self.metrics_label = tk.Label(
			sidebar,
			textvariable=self.metrics_var,
			wraplength=320,
			justify="left",
			font=SMALL_FONT,
		)
		self.metrics_label.grid(row=25, column=0, sticky="w")

		# Área de gráficos
		plot_area = tk.Frame(self.master, padx=10, pady=10)
		plot_area.grid(row=0, column=1, sticky="nsew")
		plot_area.columnconfigure(0, weight=1)
		plot_area.rowconfigure(1, weight=1)

		# Aumentar el tamaño de la figura para aprovechar la ventana más grande
		self.figure = Figure(figsize=(11, 8), dpi=110)
		self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
		toolbar = NavigationToolbar2Tk(self.canvas, plot_area)
		toolbar.update()
		toolbar.grid(row=0, column=0, sticky="ew")
		self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

		# Actualizar gráfico automáticamente al cambiar escalas
		for var in (self.time_start_var, self.time_window_var, self.y_scale_var, self.downsample_var):
			var.trace_add("write", lambda *_: self._on_controls_changed())

	# ------------------------------------------------------------------
	# Gestión de eventos
	# ------------------------------------------------------------------
	def request_file(self) -> None:
		"""Abre un diálogo para seleccionar un archivo EDF."""

		file_path = filedialog.askopenfilename(
			title="Seleccionar archivo EDF",
			filetypes=[("Archivos EDF", "*.edf"), ("Todos los archivos", "*.*")],
		)

		if not file_path:
			return

		def _load() -> None:
			try:
				data = load_edf_file(file_path)
			except (FileNotFoundError, ValueError, ImportError) as error:
				LOGGER.exception("No se pudo cargar el archivo EDF")
				messagebox.showerror("Error", str(error))
				return

			self.master.after(0, lambda: self._initialize_channels(Path(file_path), data))

		threading.Thread(target=_load, daemon=True).start()
		self.file_label.configure(text="Cargando…")

	def _initialize_channels(self, file_path: Path, channel_data: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
		self.channel_data = channel_data
		self.loaded_file = file_path
		self.file_label.configure(text=str(file_path))

		self.channel_listbox.delete(0, tk.END)
		for channel in sorted(channel_data):
			self.channel_listbox.insert(tk.END, channel)

		self._configure_time_controls()
		self._reset_predictions()

		messagebox.showinfo("Archivo cargado", f"Se cargaron {len(channel_data)} canales.")
		self._update_annotation_summary()

	def request_annotation_file(self) -> None:
		file_path = filedialog.askopenfilename(
			title="Seleccionar archivo de anotaciones",
			filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")],
		)

		if not file_path:
			return

		try:
			intervals = load_annotation_csv(file_path)
		except (FileNotFoundError, ValueError) as error:
			LOGGER.exception("No se pudo cargar el archivo de anotaciones")
			messagebox.showerror("Error", str(error))
			return

		self.annotations = intervals
		self.annotation_file = Path(file_path)
		self.annotations_enabled.set(bool(intervals))
		self._update_annotation_summary()

		if intervals:
			details = self._format_interval_details(intervals)
			messagebox.showinfo(
				"Anotaciones cargadas",
				f"Se detectaron {len(intervals)} intervalos 'seiz'.\n\n{details}",
			)
		else:
			messagebox.showinfo(
				"Anotaciones cargadas",
				"No se detectaron intervalos 'seiz' en el archivo seleccionado.",
			)

		self._try_refresh_plot()

	def clear_annotations(self) -> None:
		if not self.annotations and self.annotation_file is None:
			return

		self.annotations = []
		self.annotation_file = None
		self.annotations_enabled.set(False)
		self._update_annotation_summary()
		self._try_refresh_plot()

	def evaluate_model(self, model_type: str) -> None:
		if requests is None:
			messagebox.showerror(
				"Dependencia faltante",
				"La librería 'requests' no está instalada. Ejecuta 'pip install requests'.",
			)
			return

		if self.loaded_file is None:
			messagebox.showwarning(
				"Sin archivo EDF",
				"Carga primero un archivo EDF antes de evaluar el modelo.",
			)
			return

		model_type = model_type.lower()
		if model_type not in {"pt", "tf"}:
			messagebox.showerror("Modelo desconocido", f"Tipo de modelo no soportado: {model_type}")
			return

		endpoint = f"{API_BASE_URL}/eval_network_{model_type}"
		self.prediction_var.set("Evaluando modelo…")
		self.metrics_var.set("")
		self._set_evaluation_state(True)

		annotation_path = self.annotation_file
		edf_path = self.loaded_file

		def _worker() -> None:
			try:
				with edf_path.open("rb") as edf_stream:
					files = {
						"edf_file": (edf_path.name, edf_stream, "application/octet-stream"),
					}

					if annotation_path is not None and annotation_path.exists():
						with annotation_path.open("rb") as csv_stream:
							files["csv_file"] = (annotation_path.name, csv_stream, "text/csv")
							response = requests.post(endpoint, files=files, timeout=120)
					else:
						response = requests.post(endpoint, files=files, timeout=120)

				response.raise_for_status()
				payload = response.json()
			except FileNotFoundError as error:
				LOGGER.exception("No se pudo abrir el archivo para enviar al modelo")
				self.master.after(0, lambda: self._on_evaluation_error(str(error)))
				return
			except requests.exceptions.RequestException as error:
				LOGGER.exception("Error en la llamada al servicio de evaluación")
				self.master.after(0, lambda: self._on_evaluation_error(str(error)))
				return
			except json.JSONDecodeError as error:
				LOGGER.exception("Respuesta del servicio no es JSON válido")
				self.master.after(0, lambda: self._on_evaluation_error("Respuesta no válida del servicio (JSON)."))
				return

			self.master.after(0, lambda: self._handle_model_response(model_type, payload))

		threading.Thread(target=_worker, daemon=True).start()

	def _update_annotation_summary(self) -> None:
		if not hasattr(self, "annotation_var"):
			return

		if self.annotation_file is None:
			self.annotation_var.set("Ningún CSV cargado")
			if self.annotation_toggle is not None:
				self.annotation_toggle.configure(state=tk.DISABLED)
			self.annotations_enabled.set(False)
			return

		count = len(self.annotations)

		if self.annotation_toggle is not None:
			self.annotation_toggle.configure(state=tk.NORMAL if count else tk.DISABLED)

		if not count:
			self.annotations_enabled.set(False)

		status_word = "activas" if self.annotations_enabled.get() else "inactivas"
		interval_word = "intervalo" if count == 1 else "intervalos"
		self.annotation_var.set(
			f"{self.annotation_file.name} – {count} {interval_word} 'seiz' ({status_word})"
		)

	def _reset_predictions(self) -> None:
		self.prediction_events = []
		self.prediction_source = None
		self.predictions_enabled.set(False)
		self.metrics_var.set("")
		self._update_prediction_summary()

	def _update_prediction_summary(self) -> None:
		if not hasattr(self, "prediction_var"):
			return

		if not self.prediction_events:
			self.prediction_var.set("Sin predicciones")
			if self.prediction_toggle is not None:
				self.prediction_toggle.configure(state=tk.DISABLED)
			return

		count = len(self.prediction_events)
		source = self.prediction_source or "modelo"
		status_word = "activas" if self.predictions_enabled.get() else "inactivas"
		interval_word = "intervalo" if count == 1 else "intervalos"
		self.prediction_var.set(
			f"{source} – {count} {interval_word} ({status_word})"
		)

		if self.prediction_toggle is not None:
			self.prediction_toggle.configure(state=tk.NORMAL)

	def plot_selected(self) -> None:
		if not self.channel_data:
			messagebox.showwarning(
				"Sin datos",
				"Primero carga un archivo EDF para visualizar sus canales.",
			)
			return

		selected_indices = self.channel_listbox.curselection()
		if not selected_indices:
			messagebox.showinfo("Sin selección", "Selecciona al menos un canal para graficar.")
			return

		try:
			downsample_factor = max(1, int(self.downsample_var.get()))
		except (TypeError, ValueError):
			downsample_factor = 1
			self.downsample_var.set(1)

		channels = [self.channel_listbox.get(i) for i in selected_indices]
		self._render_plot(channels, downsample_factor)

	def _try_refresh_plot(self) -> None:
		"""Re-grafica usando la selección actual si es posible."""

		if not self.channel_data:
			return

		selected_indices = self.channel_listbox.curselection()
		if not selected_indices:
			return

		try:
			downsample_factor = max(1, int(self.downsample_var.get()))
		except (TypeError, ValueError):
			downsample_factor = 1
			self.downsample_var.set(1)

		channels = [self.channel_listbox.get(i) for i in selected_indices]
		self._render_plot(channels, downsample_factor)

	def _on_controls_changed(self) -> None:
		"""Callback cuando cambian los controles de escala; actualiza el gráfico sin mensajes."""

		self.master.after_idle(self._try_refresh_plot)

	def _on_annotation_toggle(self) -> None:
		if not self.annotations:
			self.annotations_enabled.set(False)
			return

		self._try_refresh_plot()
		self._update_annotation_summary()

	def _on_prediction_toggle(self) -> None:
		if not self.prediction_events:
			self.predictions_enabled.set(False)
			return

		self._try_refresh_plot()
		self._update_prediction_summary()

	def _render_plot(self, channels: list[str], downsample_factor: int) -> None:
		self.figure.clear()
		ax = self.figure.add_subplot(111)

		time_start, time_end = self._get_time_limits()
		y_scale = self._get_y_scale()

		for offset, channel_name in enumerate(channels):
			time_axis, samples = self.channel_data[channel_name]

			if downsample_factor > 1:
				time_axis = time_axis[::downsample_factor]
				samples = samples[::downsample_factor]

			if time_end is not None:
				mask = (time_axis >= time_start) & (time_axis <= time_end)
				if not np.any(mask):
					continue
				time_axis = time_axis[mask]
				samples = samples[mask]
			elif time_start > 0:
				mask = time_axis >= time_start
				if not np.any(mask):
					continue
				time_axis = time_axis[mask]
				samples = samples[mask]

			centered_samples = samples - np.mean(samples)
			scaled_samples = centered_samples * y_scale
			vertical_offset = offset * (np.ptp(scaled_samples) * 1.5 + 1e-6)
			ax.plot(time_axis, scaled_samples + vertical_offset, label=channel_name)

		self._render_annotations(ax, time_start, time_end)
		self._render_predictions(ax, time_start, time_end)

		ax.set_xlabel("Tiempo (s)")
		ax.set_ylabel("Amplitud (unidades EDF)")
		ax.set_title(
			f"Canales seleccionados ({len(channels)})"
			+ (f" – {self.loaded_file.name}" if self.loaded_file else "")
		)
		ax.legend(loc="upper right", fontsize="small")
		ax.grid(True, linestyle="--", alpha=0.4)

		self.figure.tight_layout()
		self.canvas.draw_idle()

	def _render_annotations(self, ax, time_start: float, time_end: float | None) -> None:
		if not self.annotations or not self.annotations_enabled.get():
			return

		visible_start = time_start
		visible_end = time_end
		legend_added = False

		for start, stop, channel, label in self.annotations:
			if stop <= visible_start:
				continue
			if visible_end is not None and start >= visible_end:
				continue

			span_start = max(start, visible_start)
			span_stop = stop if visible_end is None else min(stop, visible_end)

			if span_stop <= span_start:
				continue

			legend_label = f"Convulsión ({label})" if not legend_added else "_nolegend_"
			ax.axvspan(span_start, span_stop, color="tomato", alpha=0.35, label=legend_label)
			legend_added = True

			if channel and ax.lines:
				y_min, y_max = ax.get_ylim()
				y_pos = y_max - (y_max - y_min) * 0.05
				x_pos = span_start + (span_stop - span_start) / 2
				ax.text(
					x_pos,
					y_pos,
					f"{channel} ({label})",
					ha="center",
					va="top",
					fontsize="x-small",
					color="darkred",
					rotation=0,
				)

	def _render_predictions(self, ax, time_start: float, time_end: float | None) -> None:
		if not self.prediction_events or not self.predictions_enabled.get():
			return

		visible_start = time_start
		visible_end = time_end
		legend_added = False

		for start, stop, label, confidence in self.prediction_events:
			if stop <= visible_start:
				continue
			if visible_end is not None and start >= visible_end:
				continue

			span_start = max(start, visible_start)
			span_stop = stop if visible_end is None else min(stop, visible_end)

			if span_stop <= span_start:
				continue

			legend_label = "Predicción" if not legend_added else "_nolegend_"
			ax.axvspan(span_start, span_stop, color="royalblue", alpha=0.25, label=legend_label)
			legend_added = True

			if ax.lines:
				y_min, y_max = ax.get_ylim()
				y_pos = y_max - (y_max - y_min) * 0.12
				x_pos = span_start + (span_stop - span_start) / 2
				confidence_text = f"{confidence:.2f}" if confidence is not None else "--"
				ax.text(
					x_pos,
					y_pos,
					f"{label} ({confidence_text})",
					ha="center",
					va="top",
					fontsize="x-small",
					color="navy",
					rotation=0,
				)

	# ------------------------------------------------------------------
	# Utilidades de configuración de escala
	# ------------------------------------------------------------------
	def _configure_time_controls(self) -> None:
		self.time_max = max(
			(float(time_axis[-1]) if len(time_axis) > 0 else 0.0)
			for time_axis, _ in self.channel_data.values()
		) if self.channel_data else 0.0

		window_default = self.time_max if self.time_max > 0 else 0.0

		self.time_start_spin.configure(to=max(self.time_max, 0.0))
		self.time_window_spin.configure(
			from_=0.1,
			to=max(self.time_max, 0.5),
		)

		self.time_start_var.set(0.0)
		self.time_window_var.set(window_default if window_default > 0 else 0.0)

	def _get_time_limits(self) -> tuple[float, float | None]:
		try:
			time_start = max(0.0, float(self.time_start_var.get()))
		except (TypeError, ValueError, tk.TclError):
			time_start = 0.0
			self.time_start_var.set(0.0)

		try:
			window = max(0.0, float(self.time_window_var.get()))
		except (TypeError, ValueError, tk.TclError):
			window = 0.0
			self.time_window_var.set(0.0)

		if window <= 0 or self.time_max <= 0:
			return time_start, None

		time_end = min(time_start + window, self.time_max)
		if time_end <= time_start:
			time_start = max(0.0, min(time_start, self.time_max))
			time_end = min(time_start + window, self.time_max)

		return time_start, time_end

	def _get_y_scale(self) -> float:
		try:
			scale = float(self.y_scale_var.get())
		except (TypeError, ValueError, tk.TclError):
			scale = 1.0
			self.y_scale_var.set(1.0)

		if scale <= 0:
			scale = 1.0
			self.y_scale_var.set(1.0)

		return scale

	def _format_interval_details(self, intervals: list[tuple[float, float, str, str]], max_items: int = 10) -> str:
		lines: list[str] = []
		for idx, (start, stop, channel, label) in enumerate(intervals[:max_items], start=1):
			channel_display = channel or "Canal desconocido"
			lines.append(
				f"{idx}. {start:.2f}s – {stop:.2f}s • {channel_display} [{label}]"
			)

		remaining = len(intervals) - max_items
		if remaining > 0:
			lines.append(f"… y {remaining} intervalos adicionales.")

		return "\n".join(lines)

	def _format_prediction_details(
		self,
		intervals: list[tuple[float, float, str, float | None]],
		max_items: int = 10,
	) -> str:
		lines: list[str] = []
		for idx, (start, stop, label, confidence) in enumerate(intervals[:max_items], start=1):
			confidence_text = f"{confidence:.3f}" if confidence is not None else "--"
			lines.append(
				f"{idx}. {start:.2f}s – {stop:.2f}s • {label} (conf={confidence_text})"
			)

		remaining = len(intervals) - max_items
		if remaining > 0:
			lines.append(f"… y {remaining} intervalos adicionales.")

		return "\n".join(lines)

	def _format_metrics(self, metrics: dict[str, Any] | None) -> str:
		if not metrics:
			return "Sin métricas disponibles"

		ordered_keys = ["threshold", "precision", "recall", "f1", "accuracy"]
		lines = []
		for key in ordered_keys:
			if key in metrics:
				value = metrics[key]
				if isinstance(value, float):
					lines.append(f"{key.capitalize()}: {value:.4f}")
				else:
					lines.append(f"{key.capitalize()}: {value}")

		conf_matrix = metrics.get("confusion_matrix") if isinstance(metrics, dict) else None
		if isinstance(conf_matrix, (list, tuple)) and len(conf_matrix) == 2:
			lines.append("Matriz de confusión:")
			for row in conf_matrix:
				if isinstance(row, (list, tuple)):
					lines.append("  " + "\t".join(str(item) for item in row))

		return "\n".join(lines) if lines else "Sin métricas disponibles"

	def _set_evaluation_state(self, running: bool) -> None:
		state = tk.DISABLED if running else tk.NORMAL
		cursor = "watch" if running else ""
		for button in getattr(self, "eval_pt_button", None), getattr(self, "eval_tf_button", None):
			if button is not None:
				button.configure(state=state)

		self.master.configure(cursor=cursor)
		self.master.update_idletasks()

	def _on_evaluation_error(self, message: str) -> None:
		self._set_evaluation_state(False)
		self._update_prediction_summary()
		messagebox.showerror("Evaluación fallida", message)

	def _handle_model_response(self, model_type: str, payload: dict[str, Any]) -> None:
		self._set_evaluation_state(False)

		events_payload = payload.get("events", []) if isinstance(payload, dict) else []
		parsed_events: list[tuple[float, float, str, float | None]] = []
		for event in events_payload:
			if not isinstance(event, dict):
				continue

			try:
				start = float(event.get("start_time"))
				stop = float(event.get("stop_time"))
			except (TypeError, ValueError):
				continue

			if stop <= start:
				continue

			label = str(event.get("label") or PREDICTED_LABEL)
			confidence_raw = event.get("confidence")
			try:
				confidence = float(confidence_raw) if confidence_raw is not None else None
			except (TypeError, ValueError):
				confidence = None

			parsed_events.append((start, stop, label, confidence))

		self.prediction_events = parsed_events
		predicted_model = payload.get("model_type", model_type).upper() if isinstance(payload, dict) else model_type.upper()
		architecture = payload.get("model_architecture") if isinstance(payload, dict) else None
		self.prediction_source = (
			f"Modelo {predicted_model} ({architecture})"
			if architecture else
			f"Modelo {predicted_model}"
		)
		self.predictions_enabled.set(bool(parsed_events))
		self._update_prediction_summary()

		metrics_payload = payload.get("metrics") if isinstance(payload, dict) else None
		metrics_text = self._format_metrics(metrics_payload if isinstance(metrics_payload, dict) else None)
		self.metrics_var.set(metrics_text)

		if parsed_events:
			details = self._format_prediction_details(parsed_events)
			message_parts = [
				f"Se detectaron {len(parsed_events)} intervalos predichos.",
				details,
			]
			if metrics_payload:
				message_parts.append("\nMétricas:\n" + metrics_text)
			messagebox.showinfo("Predicciones recibidas", "\n".join(part for part in message_parts if part))
		else:
			message = "El modelo no reportó intervalos de convulsión." if payload else "El modelo no retornó datos."
			messagebox.showinfo("Predicciones recibidas", message)

		self._try_refresh_plot()


def main() -> None:
	logging.basicConfig(level=logging.INFO)

	if pyedflib is None:
		msg = "pyEDFlib no está instalado. Consulta el README para completar la instalación."
		if messagebox is not None:
			messagebox.showerror("Dependencia faltante", msg)
		else:
			LOGGER.error(msg)
		return

	if tk is None:
		LOGGER.error(
			"Tkinter no está disponible en este intérprete de Python."
			" Instala el paquete de interfaces gráficas de tu distribución."
		)
		return

	root = tk.Tk()
	EDFViewerApp(root)
	root.mainloop()


if __name__ == "__main__":
	main()
