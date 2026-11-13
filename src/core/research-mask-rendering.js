/** Build the Cornerstone volume input for a model-generated labelmap. */
export function createResearchMaskVolumeInput(volumeId, colorTransferFunction, scalarOpacity) {
  return {
    volumeId,
    // BaseVolumeViewport hides actors when visibility is omitted.
    visibility: true,
    callback: ({ volumeActor }) => {
      const property = volumeActor.getProperty();
      property.setInterpolationTypeToNearest();
      property.setRGBTransferFunction(0, colorTransferFunction);
      property.setScalarOpacity(0, scalarOpacity);
    },
  };
}
