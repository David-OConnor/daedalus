use lin_alg::{f32::Vec3 as Vec3F32, f64::Vec3};

pub fn vec3_to_f32(v: Vec3) -> Vec3F32 {
    Vec3F32::new(v.x as f32, v.y as f32, v.z as f32)
}
