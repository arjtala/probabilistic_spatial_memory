# TODO Items

## Code Duplication Issues
- [ ] Refactor duplicate HLL merging logic in `RingBuffer_merge_window` and `Tile_query`
- [ ] Remove redundant error checking in `IngestReader_open` and `ImuGpsReader_open`
- [ ] Consolidate H3 index creation code in `SpatialMemory_query` and `Tile_new`

## Consistency Issues
- [ ] Standardize precision parameter handling across all HLL operations
- [ ] Ensure consistent error return values in all HDF5 reader functions
- [ ] Align coordinate system conversions (degrees/radians) across modules

## Error Handling Issues
- [ ] Add proper error propagation in `JepaCache_lookup` instead of returning false
- [ ] Implement memory cleanup in all failure paths of `TileMap_new`
- [ ] Add bounds checking for H3 resolution parameters in `SpatialMemory_new`

## Memory Management Issues
- [ ] Fix potential memory leaks in `ImuGpsReader_interpolate_gps`
- [ ] Add proper reference counting for HLL objects in ring buffer operations
- [ ] Implement proper buffer management in `GpsTrace_push` to prevent overflow
