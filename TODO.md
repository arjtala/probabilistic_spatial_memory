# TODO Items

## Code Duplication Issues
- [ ] Refactor duplicate HLL merging logic in `RingBuffer_merge_window` and `Tile_query`
- [ ] Remove redundant error checking in `IngestReader_open` and `ImuGpsReader_open`
- [ ] Consolidate H3 index creation code in `SpatialMemory_query` and `Tile_new`
- [ ] `ring_buffer.h:20` — Abstract memory allocation and initialization logic into a utility function
- [ ] `ring_buffer.h:34` — Refactor freeing logic into a separate function to avoid repetition

## Consistency Issues
- [ ] Standardize precision parameter handling across all HLL operations
- [ ] Ensure consistent error return values in all HDF5 reader functions
- [ ] Align coordinate system conversions (degrees/radians) across modules
- [ ] `ring_buffer.h:21` — Ensure consistent naming conventions for variables and functions
- [ ] `ring_buffer.h:34` — Apply consistent patterns in other parts of the repo

## Error Handling Issues
- [ ] Add proper error propagation in `JepaCache_lookup` instead of returning false
- [ ] Implement memory cleanup in all failure paths of `TileMap_new`
- [ ] Add bounds checking for H3 resolution parameters in `SpatialMemory_new`
- [ ] `ring_buffer.h:12` — Add error handling for `realloc` calls to ensure successful allocation
- [ ] `ring_buffer.h:30` — Implement proper error checking for critical operations like `H5Dopen`

## Memory Management Issues
- [ ] Fix potential memory leaks in `ImuGpsReader_interpolate_gps`
- [ ] Add proper reference counting for HLL objects in ring buffer operations
- [ ] Implement proper buffer management in `GpsTrace_push` to prevent overflow
- [ ] `ring_buffer.h:12` — Ensure proper error handling in `realloc` to avoid allocation failures
- [ ] `ring_buffer.h:13` — Add bounds checking before accessing array elements
