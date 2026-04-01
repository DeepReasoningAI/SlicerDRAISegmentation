import os
import uuid
import json
import tempfile
import logging
from typing import Optional

import vtk
import qt
import ctk
import slicer
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleWidget,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
)
from slicer.util import VTKObservationMixin

try:
    import requests

    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


AVAILABLE_MODELS = [
    ("Spine and Pelvis", "spine"),
    ("Aorta-Iliac-Femoral", "aif"),
]

DEFAULT_SERVER_URL = "http://localhost:8000"
SETTINGS_KEY_PREFIX = "DRAISegmentation"
POLL_INTERVAL_MS = 5000
JOB_TIMEOUT_MS = 600000  # 10 minutes


# ---------------------------------------------------------------------------
# Module descriptor
# ---------------------------------------------------------------------------
class DRAISegmentation(ScriptedLoadableModule):

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = "DRAI Segmentation"
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["DRAI Team (Deep Reasoning AI)"]
        self.parent.helpText = (
            "AI-powered segmentation for Spine/Pelvis (CT) and Aorta–Iliac–Femoral Arteries (CTA). "
            "Upload CT volumes to the DRAI cloud and receive high-quality, multi-label masks directly in 3D Slicer within minutes."
            "Built to handle complex cases—including bypass grafts, metal implants, and challenging pathology—with reliable accuracy."
            "All data is securely processed and deleted immediately after inference. Questions? Contact support@deepreasoningai.com."
            'See the <a href="https://github.com/DeepReasoningAI/SlicerDRAISegmentation">'
            "documentation</a> for more information."
        )
        self.parent.acknowledgementText = (
            "Developed by Deep Reasoning AI (DRAI). "
            "All uploaded images are deleted from the server immediately "
            "after segmentation is complete. No data is retained." 
            "Questions? Contact support@deepreasoningai.com."
        )


# ---------------------------------------------------------------------------
# Widget (UI)
# ---------------------------------------------------------------------------
class DRAISegmentationWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)
        self.logic: Optional[DRAISegmentationLogic] = None
        self._pollTimer: Optional[qt.QTimer] = None
        self._currentJobId: Optional[str] = None
        self._elapsedMs = 0

    # ---- setup -------------------------------------------------------------

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        self.logic = DRAISegmentationLogic()
        self.logic.ensureDeviceRegistered()

        self._buildUI()
        self._connectSignals()
        self._setUIIdle()

    def _buildUI(self):
        # --- Server Configuration (collapsible, advanced) ---
        serverCollapsible = ctk.ctkCollapsibleButton()
        serverCollapsible.text = "Server Configuration (Advanced)"
        serverCollapsible.collapsed = True
        self.layout.addWidget(serverCollapsible)
        serverLayout = qt.QFormLayout(serverCollapsible)

        self.serverUrlEdit = qt.QLineEdit()
        self.serverUrlEdit.text = self.logic.getServerUrl()
        self.serverUrlEdit.toolTip = "Base URL of the DRAI segmentation server"
        serverLayout.addRow("Server URL:", self.serverUrlEdit)

        self.deviceIdLabel = qt.QLineEdit()
        self.deviceIdLabel.text = self.logic.getDeviceId()
        self.deviceIdLabel.readOnly = True
        self.deviceIdLabel.toolTip = "Auto-generated device identifier (for support)"
        serverLayout.addRow("Device ID:", self.deviceIdLabel)

        # --- Inputs ---
        inputsCollapsible = ctk.ctkCollapsibleButton()
        inputsCollapsible.text = "Inputs"
        self.layout.addWidget(inputsCollapsible)
        inputsLayout = qt.QFormLayout(inputsCollapsible)

        self.volumeSelector = slicer.qMRMLNodeComboBox()
        self.volumeSelector.nodeTypes = ["vtkMRMLScalarVolumeNode"]
        self.volumeSelector.selectNodeUponCreation = True
        self.volumeSelector.addEnabled = False
        self.volumeSelector.removeEnabled = False
        self.volumeSelector.noneEnabled = True
        self.volumeSelector.showHidden = False
        self.volumeSelector.showChildNodeTypes = False
        self.volumeSelector.setMRMLScene(slicer.mrmlScene)
        self.volumeSelector.toolTip = "Select the input volume for segmentation"
        inputsLayout.addRow("Input Volume:", self.volumeSelector)

        self.modelSelector = qt.QComboBox()
        for displayName, _ in AVAILABLE_MODELS:
            self.modelSelector.addItem(displayName)
        self.modelSelector.toolTip = "Select the segmentation model to apply"
        inputsLayout.addRow("Model:", self.modelSelector)

        # --- Run ---
        runCollapsible = ctk.ctkCollapsibleButton()
        runCollapsible.text = "Run Segmentation"
        self.layout.addWidget(runCollapsible)
        runLayout = qt.QVBoxLayout(runCollapsible)

        self.runButton = qt.QPushButton("Run Segmentation")
        self.runButton.toolTip = "Upload the selected volume and run segmentation"
        self.runButton.enabled = False
        runLayout.addWidget(self.runButton)

        self.statusLabel = qt.QLabel("")
        runLayout.addWidget(self.statusLabel)

        self.progressBar = qt.QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.visible = False
        runLayout.addWidget(self.progressBar)

        self.cancelButton = qt.QPushButton("Cancel")
        self.cancelButton.visible = False
        runLayout.addWidget(self.cancelButton)

        # --- Data Privacy Notice ---
        privacyCollapsible = ctk.ctkCollapsibleButton()
        privacyCollapsible.text = "Data Privacy Notice"
        self.layout.addWidget(privacyCollapsible)
        privacyLayout = qt.QVBoxLayout(privacyCollapsible)

        privacyText = qt.QLabel(
            '<p style="color: #555;">'
            "<b>Important:</b> By clicking <i>Run Segmentation</i>, your "
            "medical image will be uploaded to the DRAI server for processing. "
            "All uploaded images are <b>deleted from our server immediately</b> "
            "after segmentation is complete. <b>No data is retained.</b>"
            "</p>"
        )
        privacyText.wordWrap = True
        privacyLayout.addWidget(privacyText)

        self.layout.addStretch(1)

    def _connectSignals(self):
        self.volumeSelector.connect("currentNodeChanged(vtkMRMLNode*)", self._onInputChanged)
        self.runButton.connect("clicked(bool)", self._onRunClicked)
        self.cancelButton.connect("clicked(bool)", self._onCancelClicked)
        self.serverUrlEdit.connect("editingFinished()", self._onServerUrlChanged)

    # ---- UI state helpers ---------------------------------------------------

    def _setUIIdle(self):
        self._updateRunButtonState()
        self.progressBar.visible = False
        self.progressBar.value = 0
        self.cancelButton.visible = False
        self.statusLabel.text = ""
        self.volumeSelector.enabled = True
        self.modelSelector.enabled = True

    def _setUIBusy(self, statusText=""):
        self.runButton.enabled = False
        self.volumeSelector.enabled = False
        self.modelSelector.enabled = False
        self.progressBar.visible = True
        self.cancelButton.visible = True
        self.statusLabel.text = statusText

    def _updateRunButtonState(self):
        self.runButton.enabled = self.volumeSelector.currentNode() is not None

    # ---- signal handlers ----------------------------------------------------

    def _onInputChanged(self, node):
        self._updateRunButtonState()

    def _onServerUrlChanged(self):
        url = self.serverUrlEdit.text.strip()
        if url:
            self.logic.setServerUrl(url)

    def _onRunClicked(self, _checked):
        volumeNode = self.volumeSelector.currentNode()
        if volumeNode is None:
            slicer.util.errorDisplay("Please select an input volume.")
            return

        if not self._showConsentDialog():
            return

        modelIndex = self.modelSelector.currentIndex
        _, modelKey = AVAILABLE_MODELS[modelIndex]

        self._setUIBusy("Exporting volume...")
        slicer.app.processEvents()

        try:
            tempPath = self.logic.exportVolume(volumeNode)
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to export volume:\n{exc}")
            self._setUIIdle()
            return

        self._setUIBusy("Uploading to server...")
        slicer.app.processEvents()

        try:
            jobId = self.logic.submitJob(tempPath, modelKey)
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to submit job:\n{exc}")
            self._setUIIdle()
            return

        self._currentJobId = jobId
        self._elapsedMs = 0
        self._setUIBusy(f"Job submitted (ID: {jobId}). Waiting for results...")
        self._startPolling()

    def _onCancelClicked(self, _checked):
        self._stopPolling()
        self._currentJobId = None
        self._setUIIdle()
        self.statusLabel.text = "Cancelled by user."

    # ---- consent dialog -----------------------------------------------------

    def _showConsentDialog(self) -> bool:
        msgBox = qt.QMessageBox()
        msgBox.setWindowTitle("Data Privacy Consent")
        msgBox.setIcon(qt.QMessageBox.Warning)
        msgBox.setText(
            "Your medical image will be uploaded to the DRAI server for "
            "AI segmentation processing.\n\n"
            "All uploaded images are deleted from the server immediately "
            "after segmentation is complete. No data is retained.\n\n"
            "Do you agree to proceed?"
        )
        agreeButton = msgBox.addButton("I Agree", qt.QMessageBox.AcceptRole)
        msgBox.addButton("Cancel", qt.QMessageBox.RejectRole)
        msgBox.exec_()
        return msgBox.clickedButton() == agreeButton

    # ---- polling ------------------------------------------------------------

    def _startPolling(self):
        self._pollTimer = qt.QTimer()
        self._pollTimer.setInterval(POLL_INTERVAL_MS)
        self._pollTimer.connect("timeout()", self._pollStatus)
        self._pollTimer.start()

    def _stopPolling(self):
        if self._pollTimer is not None:
            self._pollTimer.stop()
            self._pollTimer = None

    def _pollStatus(self):
        if self._currentJobId is None:
            self._stopPolling()
            return

        self._elapsedMs += POLL_INTERVAL_MS
        if self._elapsedMs > JOB_TIMEOUT_MS:
            self._stopPolling()
            slicer.util.errorDisplay(
                "Segmentation timed out. Please try again or contact support."
            )
            self._setUIIdle()
            return

        try:
            status, progress = self.logic.getJobStatus(self._currentJobId)
        except Exception as exc:
            logging.warning(f"Status poll failed: {exc}")
            self.statusLabel.text = "Connection issue -- retrying..."
            return

        if status == "queued":
            self.statusLabel.text = "Queued on server..."
            self.progressBar.value = 0
        elif status == "processing":
            pct = progress if progress is not None else 0
            self.progressBar.value = pct
            self.statusLabel.text = f"Processing... {pct}%"
        elif status == "completed":
            self._stopPolling()
            self._downloadAndLoadResult()
        elif status == "failed":
            self._stopPolling()
            slicer.util.errorDisplay(
                "Segmentation failed on the server. Please try again or "
                "contact support."
            )
            self._setUIIdle()
        else:
            self.statusLabel.text = f"Unknown status: {status}"

    def _downloadAndLoadResult(self):
        self._setUIBusy("Downloading segmentation result...")
        slicer.app.processEvents()

        try:
            resultPath = self.logic.downloadResult(self._currentJobId)
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to download result:\n{exc}")
            self._setUIIdle()
            return

        self._setUIBusy("Loading segmentation into scene...")
        slicer.app.processEvents()

        try:
            volumeNode = self.volumeSelector.currentNode()
            self.logic.loadSegmentation(resultPath, volumeNode)
        except Exception as exc:
            slicer.util.errorDisplay(f"Failed to load segmentation:\n{exc}")
            self._setUIIdle()
            return

        self._currentJobId = None
        self._setUIIdle()
        self.statusLabel.text = "Segmentation complete!"
        slicer.util.infoDisplay("Segmentation loaded successfully.")

    # ---- cleanup ------------------------------------------------------------

    def cleanup(self):
        self._stopPolling()


# ---------------------------------------------------------------------------
# Logic
# ---------------------------------------------------------------------------
class DRAISegmentationLogic(ScriptedLoadableModuleLogic):

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._settings = qt.QSettings()

    # ---- device registration ------------------------------------------------

    def getDeviceId(self) -> str:
        key = f"{SETTINGS_KEY_PREFIX}/DeviceId"
        deviceId = self._settings.value(key, "")
        if not deviceId:
            deviceId = str(uuid.uuid4())
            self._settings.setValue(key, deviceId)
            self._settings.sync()
        return deviceId

    def getServerUrl(self) -> str:
        key = f"{SETTINGS_KEY_PREFIX}/ServerUrl"
        return self._settings.value(key, DEFAULT_SERVER_URL)

    def setServerUrl(self, url: str):
        key = f"{SETTINGS_KEY_PREFIX}/ServerUrl"
        self._settings.setValue(key, url)
        self._settings.sync()

    def _headers(self) -> dict:
        return {"X-Device-ID": self.getDeviceId()}

    def ensureDeviceRegistered(self):
        key = f"{SETTINGS_KEY_PREFIX}/DeviceRegistered"
        if self._settings.value(key, "false") == "true":
            return

        deviceId = self.getDeviceId()
        url = f"{self.getServerUrl()}/api/v1/register"

        try:
            if HAS_REQUESTS:
                resp = requests.post(
                    url,
                    json={"device_id": deviceId},
                    headers=self._headers(),
                    timeout=15,
                )
                resp.raise_for_status()
            else:
                self._postJson(url, {"device_id": deviceId})

            self._settings.setValue(key, "true")
            self._settings.sync()
            logging.info("Device registered with DRAI server.")
        except Exception as exc:
            logging.warning(
                f"Device registration failed (will retry next launch): {exc}"
            )

    # ---- volume export ------------------------------------------------------

    def exportVolume(self, volumeNode) -> str:
        tmpDir = tempfile.mkdtemp(prefix="drai_")
        filename = "volume.nii.gz"
        filepath = os.path.join(tmpDir, filename)
        success = slicer.util.saveNode(volumeNode, filepath)
        if not success:
            raise RuntimeError(f"Could not save volume to {filepath}")
        return filepath

    # ---- job submission -----------------------------------------------------

    def submitJob(self, niftiPath: str, model: str) -> str:
        url = f"{self.getServerUrl()}/api/v1/jobs"

        if HAS_REQUESTS:
            with open(niftiPath, "rb") as f:
                resp = requests.post(
                    url,
                    files={"file": (os.path.basename(niftiPath), f, "application/gzip")},
                    data={"model": model},
                    headers=self._headers(),
                    timeout=300,
                )
            resp.raise_for_status()
            data = resp.json()
        else:
            data = self._postMultipart(url, niftiPath, model)

        jobId = data.get("job_id")
        if not jobId:
            raise RuntimeError(f"Server did not return a job_id: {data}")
        return jobId

    # ---- status polling -----------------------------------------------------

    def getJobStatus(self, jobId: str):
        url = f"{self.getServerUrl()}/api/v1/jobs/{jobId}/status"

        if HAS_REQUESTS:
            resp = requests.get(url, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
        else:
            data = self._getJson(url)

        status = data.get("status", "unknown")
        progress = data.get("progress")
        return status, progress

    # ---- result download ----------------------------------------------------

    def downloadResult(self, jobId: str) -> str:
        url = f"{self.getServerUrl()}/api/v1/jobs/{jobId}/result"
        tmpDir = tempfile.mkdtemp(prefix="drai_result_")
        outPath = os.path.join(tmpDir, "segmentation.nii.gz")

        if HAS_REQUESTS:
            resp = requests.get(url, headers=self._headers(), timeout=300, stream=True)
            resp.raise_for_status()
            with open(outPath, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
        else:
            self._downloadFile(url, outPath)

        return outPath

    # ---- load segmentation into Slicer --------------------------------------

    def loadSegmentation(self, niftiPath: str, referenceVolumeNode=None):
        labelmapNode = slicer.util.loadLabelVolume(niftiPath, {"name": "DRAI_Segmentation"})

        segmentationNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentationNode.SetName("DRAI Segmentation Result")

        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapNode, segmentationNode
        )

        slicer.mrmlScene.RemoveNode(labelmapNode)

        if referenceVolumeNode is not None:
            segmentationNode.SetReferenceImageGeometryParameterFromVolumeNode(referenceVolumeNode)

        segmentationNode.CreateClosedSurfaceRepresentation()

        slicer.util.setSliceViewerLayers(
            background=referenceVolumeNode if referenceVolumeNode else None
        )

        layoutManager = slicer.app.layoutManager()
        threeDWidget = layoutManager.threeDWidget(0)
        if threeDWidget:
            threeDView = threeDWidget.threeDView()
            threeDView.resetFocalPoint()

        return segmentationNode

    # ---- urllib fallback helpers ---------------------------------------------

    def _postJson(self, url: str, body: dict) -> dict:
        import urllib.request
        import urllib.error

        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={**self._headers(), "Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _getJson(self, url: str) -> dict:
        import urllib.request

        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _postMultipart(self, url: str, filepath: str, model: str) -> dict:
        import urllib.request

        boundary = uuid.uuid4().hex
        lines = []

        lines.append(f"--{boundary}".encode())
        lines.append(b'Content-Disposition: form-data; name="model"')
        lines.append(b"")
        lines.append(model.encode())

        with open(filepath, "rb") as f:
            filedata = f.read()

        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="file"; filename="{os.path.basename(filepath)}"'.encode()
        )
        lines.append(b"Content-Type: application/gzip")
        lines.append(b"")
        lines.append(filedata)
        lines.append(f"--{boundary}--".encode())

        body = b"\r\n".join(lines)

        req = urllib.request.Request(
            url,
            data=body,
            headers={
                **self._headers(),
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _downloadFile(self, url: str, outPath: str):
        import urllib.request

        req = urllib.request.Request(url, headers=self._headers(), method="GET")
        with urllib.request.urlopen(req, timeout=300) as resp:
            with open(outPath, "wb") as f:
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    f.write(chunk)


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
class DRAISegmentationTest(ScriptedLoadableModuleTest):

    def setUp(self):
        slicer.mrmlScene.Clear()

    def runTest(self):
        self.setUp()
        self.test_ModuleLoads()

    def test_ModuleLoads(self):
        self.delayDisplay("Starting module load test")
        logic = DRAISegmentationLogic()
        self.assertIsNotNone(logic.getDeviceId())
        self.assertTrue(len(logic.getDeviceId()) > 0)
        self.delayDisplay("Module load test passed")
