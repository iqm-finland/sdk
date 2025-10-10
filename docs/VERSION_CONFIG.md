# Adding New OS Versions

To add a new OS version (e.g., IQM OS 5.0), you only need to edit one file: `docs/src/configs.ts`

## Steps:

1. **Update the VersionType**:
   ```typescript
   export type VersionType = 'resonance' | 'os4.1' | 'os4.2' | 'os5.0';
   ```

2. **Add the new version configuration**:
   ```typescript
   {
     id: 'os5.0',
     label: 'IQM OS 5.0',
     pathPrefix: './sdk5_0/',
     description: 'You are viewing documentation for IQM OS 5.0. Some packages may not be available in this version.',
     packages: [
       'iqm-pulla',
       'iqm-client',
       'iqm-pulse',
       // Add other packages available in OS 5.0
     ]
   }
   ```

3. **Create the SDK file** (for the build process):
   Create `sdk5_0.txt` in the root with the package versions:
   ```
   iqm-pulla[qiskit,qir]==12.0
   iqm-client[qiskit,cirq,cli]==35.0
   iqm-pulse==15.0
   ```

That's it! The build system, UI, and URL handling will automatically support the new version.

## Configuration Properties:

- **id**: Unique identifier used in URLs and internal logic
- **label**: Display name shown in the UI
- **pathPrefix**: URL path where the documentation will be served
- **description**: Optional warning/info message shown when this version is selected
- **packages**: Array of package names available in this version

## Automatic Features:

- Version selector buttons are generated automatically
- URL state persistence (e.g., `?version=os5.0`)
- Package filtering (only shows packages available in the selected version)
- Build system automatically processes `sdk*_*.txt` files
- Warning messages for non-default versions