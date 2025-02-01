use lin_alg::f64::Vec3;

pub fn vec3_to_f32(v: Vec3) -> lin_alg::f32::Vec3 {
    lin_alg::f32::Vec3::new(v.x as f32, v.y as f32, v.z as f32)
}
