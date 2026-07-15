import { useState, useEffect } from "react";
import { createPortal } from "react-dom";


const Tooltip = ({ content, children }: { content: JSX.Element, children: JSX.Element }) => {
    const [visible, setVisible] = useState(false);
    const [tooltipContainer, setTooltipContainer] = useState<HTMLElement | null>(null);
    const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 });

    useEffect(() => {
        const container = document.createElement('div');
        document.body.appendChild(container);
        setTooltipContainer(container);

        return () => {
            document.body.removeChild(container);
        };
    }, []);

    const handleMouseMove = (event: React.MouseEvent) => {
        setMousePosition({ x: event.clientX, y: event.clientY });
    };

    if (!tooltipContainer) return null;

    return (
        <div className="relative inline-block" onMouseMove={handleMouseMove}>
            {children}
            <span
                className="tooltip-trigger ml-1 cursor-pointer"
                onMouseEnter={() => setVisible(true)}
                onMouseLeave={() => setVisible(false)}
                onClick={() => setVisible(!visible)}
            >
                &#9432;
            </span>
            {visible && createPortal(
                <div
                    className="fixed p-2 bg-gray-700 text-white text-sm rounded shadow-lg z-50"
                    style={{ top: mousePosition.y + 10, left: mousePosition.x + 10 }}
                >
                    {content}
                </div>,
                tooltipContainer
            )}
        </div>
    );
};


const gateFeatures = [
    {
        name: "Higher energy states (resonator)", // name of the feature
        qccsw: "3.1.0", // first version of QCCSW supporting the feature
        qiskit: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/deneb/Deneb_Unlocked_Resonator.ipynb"
        }, // Qiskit tutorial link or true if supported
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.move_gate_validation"
        },
    },
    {
        name: "Mid-circuit measurements",
        qccsw: "3.1.0",
        qiskit: true,
        qrisp: true,
        cirq: true,
        cudaq: true,
    },
    {
        name: "Classically controlled gates",
        qccsw: "3.1.0",
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#classically-controlled-gates"
        },
        qrisp: false,
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/user_guide_cirq.html#classical-control"
        },
        cudaq: false,
    },
    {
        name: <Tooltip content={"Increase throughput by batching circuits that all read out the same qubits."}>Batched execution </Tooltip>,

        qccsw: "1.0.0",
        qiskit: true,
        cirq: true,
        cudaq: true,
        qrisp: true,
    },
    {
        name: "Dynamical decoupling",
        qccsw: "3.3.0",
        qrisp: true,
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html"
        },
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html"
        },
    },
    {
        name: <div style={{ display: "flex" }}> <Tooltip content={"Using a secondary detection event to confirm the successful preparation or measurement of a quantum state."
        }> Heralding</Tooltip></div>
        , qccsw: "1.0.0",
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.heralding_mode"
        },
        qrisp: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.heralding_mode"
        },
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.heralding_mode"
        },
    },
    {
        name: "Benchmarking tools",
        qccsw: "-",
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-benchmarks/"
        }
    },
    {
        name: "Simulated backend",
        qccsw: "-",
        qiskit: true,
        cirq: "-",
        cudaq: "-",
        qrisp: "-",
    },
    {
        name: "Compilation check",
        qccsw: "1.0.0",
        qiskit: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/garnet/GarnetAlgorithmsChecker.ipynb"
        },
        cirq: true,
        cudaq: true,
        qrisp: true,
    },
    {
        name: "Resetting qubits",
        qccsw: "3.2.0",
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-client/user_guide_qiskit.html#resetting-qubits"
        },
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/user_guide_cirq.html#resetting-qubits"
        },
        cudaq: false,
        qrisp: true,
    },
    {
        name: <Tooltip content="The qubits are actively reset once more using conditional pulses feedback loops before circuit execution.">Automated active reset</Tooltip>,
        qccsw: "3.3.0",
        qrisp: true,
        qiskit: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.active_reset_cycles"
        },
        cirq: {
            tutorial: "https://docs.iqm.tech/iqm-client/api/iqm.iqm_client.models.CircuitCompilationOptions.html#iqm.iqm_client.models.CircuitCompilationOptions.active_reset_cycles"
        },
    },
    {
        name: "Programmatically retrieve calibration data (Resonance)",
        qccsw: "-",
        qiskit: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/general/RetrieveCalibrationData.ipynb"
        },
        cirq: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/general/RetrieveCalibrationData.ipynb"
        },
        cudaq: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/general/RetrieveCalibrationData.ipynb"
        },
        qrisp: {
            tutorial: "https://www.iqmacademy.com/notebookViewer/?path=/notebooks/iqm/general/RetrieveCalibrationData.ipynb"
        },
    },
    {
        name: "Programmatically retrieve calibration data",
        qccsw: "-",
        qiskit: true,
        cirq: true,
        cudaq: true,
        qrisp: true,
    },
    {
        name: "MOVE operation support",
        qccsw: "3.0.0",
        qiskit: {
            tutorial: "https://www.iqmacademy.com/learn/deneb/01-move/"
        },
        cirq: {
            tutorial: "https://www.iqmacademy.com/learn/deneb/01-move/"
        }
    },
]

const pulseFeatures = [
    {
        name: "Ready-made experiments",
        resonance: false,
        onprem: true,
    },
    {
        name: "Custom calibrations",
        resonance: true,
        onprem: true,
    },
    {
        name: "Custom gates",
        resonance: true,
        onprem: true,
    },
    {
        name: "Pulse Schedule viewer",
        resonance: true,
        onprem: true,
    },
    {
        name: "Custom compiler stages",
        resonance: true,
        onprem: true,
    },
    {
        name: "Calibration workflows",
        resonance: false,
        onprem: true,
    }
]

export { gateFeatures, pulseFeatures, Tooltip };
