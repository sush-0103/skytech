import struct
import json
import os

def create_gray_drone_glb(input_path, output_path):
    with open(input_path, 'rb') as f:
        # 12-byte header
        magic, version, total_length = struct.unpack('<4sII', f.read(12))
        if magic != b'glTF':
            raise ValueError("Not a valid glTF binary file")
        
        # Chunk 0: JSON
        chunk0_len, chunk0_type = struct.unpack('<I4s', f.read(8))
        json_bytes = f.read(chunk0_len)
        data = json.loads(json_bytes.decode('utf-8'))
        
        # Remaining file is chunk 1 (BIN buffer)
        bin_chunk_header = f.read(8)
        bin_data = f.read()

    # Map meshes to node names
    nodes = data.get('nodes', [])
    meshes = data.get('meshes', [])
    node_mesh_map = {}
    for n in nodes:
        if 'mesh' in n:
            node_mesh_map[n['mesh']] = n.get('name', '').lower()

    # Define PBR palettes
    light_gray_pbr = {
        "baseColorFactor": [0.8235, 0.8431, 0.8745, 1.0], # #d2d7df Light Gray
        "metallicFactor": 0.24,
        "roughnessFactor": 0.38
    }
    
    dark_carbon_pbr = {
        "baseColorFactor": [0.153, 0.153, 0.165, 1.0], # #27272a Dark Carbon
        "metallicFactor": 0.1,
        "roughnessFactor": 0.5
    }
    
    camera_pbr = {
        "baseColorFactor": [0.69, 0.72, 0.765, 1.0],
        "metallicFactor": 0.45,
        "roughnessFactor": 0.25
    }

    materials = data.get('materials', [])
    for m_idx, mesh in enumerate(meshes):
        node_name = node_mesh_map.get(m_idx, '')
        for prim in mesh.get('primitives', []):
            mat_idx = prim.get('material')
            if mat_idx is not None and mat_idx < len(materials):
                mat = materials[mat_idx]
                mat['name'] = f"GrayDrone_{node_name}"
                
                if 'prop_' in node_name:
                    mat['pbrMetallicRoughness'] = dict(dark_carbon_pbr)
                elif 'camera' in node_name and 'screw' not in node_name:
                    mat['pbrMetallicRoughness'] = dict(camera_pbr)
                else:
                    mat['pbrMetallicRoughness'] = dict(light_gray_pbr)

    # Encode updated JSON
    new_json_bytes = json.dumps(data, separators=(',', ':')).encode('utf-8')
    
    # Pad JSON chunk to 4-byte boundary with spaces (0x20)
    padding_len = (4 - (len(new_json_bytes) % 4)) % 4
    new_json_bytes += b' ' * padding_len
    new_chunk0_len = len(new_json_bytes)

    # Calculate new total length
    new_total_length = 12 + 8 + new_chunk0_len + len(bin_chunk_header) + len(bin_data)

    # Write output GLB
    with open(output_path, 'wb') as out:
        # Header
        out.write(struct.pack('<4sII', magic, version, new_total_length))
        # Chunk 0
        out.write(struct.pack('<I4s', new_chunk0_len, b'JSON'))
        out.write(new_json_bytes)
        # Chunk 1
        out.write(bin_chunk_header)
        out.write(bin_data)

    print(f"Successfully generated {output_path} ({new_total_length} bytes)")

if __name__ == '__main__':
    create_gray_drone_glb('source/fly.glb', 'gray_drone.glb')
    create_gray_drone_glb('source/fly.glb', 'source/gray_drone.glb')
