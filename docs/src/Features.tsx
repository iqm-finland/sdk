import React from "react";
import { gateFeatures, pulseFeatures, Tooltip } from "./featurelist";

interface FeatureCellProps {
    framework: any;
}

const FeatureCell: React.FC<FeatureCellProps> = ({ framework }) => {
    return (
        <div>
            {framework ? (
                <>
                    {typeof framework === "string" ? (
                        framework
                    ) : (
                        <>&#9989;</>
                    )}
                </>
            ) : (
                <>❌</>
            )}{" "}
            {framework && framework.tutorial && (
                <>

                        <a
                            href={framework.tutorial}
                            target="_blank"
                            rel="noreferrer"
                        >
                            🔍
                        </a>
                </>
            )}
        </div>
    );
};

const Features: React.FC = () => {
    return (
        <>
            <main>
                <div className="container mx-auto">

                    <div className="my-4">
                        <p>
                            IQM Quantum Computers support multiple quantum computing frameworks with support for varying features.
                            Below's list provides an overview of the features supported by IQM Quantum Computers.
                        </p>
                        <p className="mt-4">If the feature you require is not listed here, it might not mean it is not supported. Please contact us at <a href="mailto:support@meetiqm.com">IQM Support</a>.</p>
                    </div>

                    <h2 className="text-2xl font-semibold mt-8">Gate-based access</h2>
                    <p>Click on the 🔍 to access more information.</p>
                    <div className="overflow-x-auto pt-4 pb-4">
                        <table className="min-w-full bg-white rounded-xl">
                            <thead>
                                <tr>
                                    <th className="py-2 px-4 border-b">FEATURE</th>
                                    <th className="py-2 px-4 border-b">
                                        <div className="flex items-center">
                                           <Tooltip content={<>Earliest version of the Quantum Computer Software Stack needed to support this feature.</>}>IQM OS</Tooltip>
                                        </div>
                                    </th>
                                    <th className="py-2 px-4 border-b"><Tooltip content="For supported versions check the quantum computer detail page.">Qiskit</Tooltip></th>
                                    <th className="py-2 px-4 border-b"><Tooltip content="Supported in Resonance via Qiskit-on-IQM.">qrisp</Tooltip></th>
                                    <th className="py-2 px-4 border-b"><Tooltip content="For supported versions check the quantum computer detail page.">Cirq</Tooltip></th>
                                    <th className="py-2 px-4 border-b">CUDA-Q</th>

                                </tr>
                            </thead>
                            <tbody>
                                {gateFeatures
                                    .sort((a, b) => b.qccsw.localeCompare(a.qccsw))
                                    .map((feature, index) => (
                                        <tr key={index}>
                                            <td className="py-2 px-4 border-b">{feature.name}</td>
                                            <td className="py-2 px-4 border-b">{typeof feature.qccsw === "string" && /\d/.test(feature.qccsw[0]) && <>&ge;</>} {feature.qccsw}</td>
                                            <td className="py-2 px-4 border-b">
                                                <FeatureCell framework={feature.qiskit} />
                                            </td>
                                            <td className="py-2 px-4 border-b">
                                                <FeatureCell framework={feature.qrisp} />
                                            </td>
                                            <td className="py-2 px-4 border-b">
                                                <FeatureCell framework={feature.cirq} />
                                            </td>
                                            <td className="py-2 px-4 border-b">
                                                <FeatureCell framework={feature.cudaq} />
                                            </td>
                                        </tr>
                                    ))}
                            </tbody>
                        </table>

                    </div>

                    <div className="my-8">
                        <h2 className="text-2xl font-semibold">Pulse-based access</h2>
                        <div className="overflow-x-auto">
                            <table className="min-w-full bg-white rounded-xl mt-4">
                                <thead>
                                    <tr>
                                        <th className="py-2 px-4 border-b">FEATURE</th>
                                        <th className="py-2 px-4 border-b">RESONANCE</th>
                                        <th className="py-2 px-4 border-b">ON-PREMISE DEVICES</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {pulseFeatures
                                        .sort((a, b) => {
                                            const nameA = typeof a.name === "string" ? a.name : a.name.props.children[0];
                                            const nameB = typeof b.name === "string" ? b.name : b.name.props.children[0];
                                            return nameA.localeCompare(nameB);
                                        })
                                        .map((feature, index) => (
                                            <tr key={index}>
                                                <td className="py-2 px-4 border-b">{feature.name}</td>
                                                <td className="py-2 px-4 border-b justify-items-center">
                                                    <FeatureCell framework={feature.resonance} />
                                                </td>
                                                <td className="py-2 px-4 border-b justify-items-center">
                                                    <FeatureCell framework={feature.onprem} />
                                                </td>
                                            </tr>
                                        ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            </main>
        </>
    );
};

export default Features;
