export type VersionType = 'resonance' | 'os4.1' | 'os4.2';

export interface VersionConfig {
  id: VersionType;
  label: string;
  pathPrefix: string;
  description?: string;
  packages: string[];
}

export const versionConfigs: VersionConfig[] = [
  {
    id: 'resonance',
    label: 'IQM OS 4.2 (Resonance)',
    pathPrefix: './',
    packages: [
      'iqm-data-definitions',
      'iqm-exa-common',
      'iqm-station-control-client',
      'iqm-pulse',
      'iqm-pulla',
      'iqm-client',
      'iqm-qaoa'
    ]
  },
  {
    id: 'os4.1',
    label: 'IQM OS 4.1',
    pathPrefix: './sdk4_1/',
    description: 'You are viewing documentation for IQM OS 4.1. This version applies for on-premises installations.',
    packages: [
      'iqm-data-definitions',
      'iqm-exa-common',
      'iqm-station-control-client',
      'iqm-pulse',
      'iqm-pulla',
      'iqm-client',
      'iqm-qaoa'
    ]
  }
];