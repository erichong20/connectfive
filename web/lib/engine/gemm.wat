;; out[r][o] = sum_k patches[r][k] * kernelT[o][k], with f32x4 SIMD.
;; K (row length) must be a multiple of 4 and cout a multiple of 4.
;; Build with `npm run build:wasm` (writes gemm-wasm.ts).
(module
  (memory (export "memory") 16)
  (func (export "gemm")
    (param $p i32) (param $k i32) (param $out i32)
    (param $rows i32) (param $K i32) (param $cout i32)
    (local $r i32) (local $o i32) (local $kk i32) (local $stride i32)
    (local $prow i32) (local $k0 i32) (local $k1 i32) (local $k2 i32) (local $k3 i32)
    (local $optr i32) (local $pv v128)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128)
    (local.set $stride (i32.shl (local.get $K) (i32.const 2)))
    (local.set $optr (local.get $out))
    (block $rows_done
      (loop $rows_loop
        (br_if $rows_done (i32.ge_u (local.get $r) (local.get $rows)))
        (local.set $prow (i32.add (local.get $p) (i32.mul (local.get $r) (local.get $stride))))
        (local.set $o (i32.const 0))
        (block $out_done
          (loop $out_loop
            (br_if $out_done (i32.ge_u (local.get $o) (local.get $cout)))
            (local.set $k0 (i32.add (local.get $k) (i32.mul (local.get $o) (local.get $stride))))
            (local.set $k1 (i32.add (local.get $k0) (local.get $stride)))
            (local.set $k2 (i32.add (local.get $k1) (local.get $stride)))
            (local.set $k3 (i32.add (local.get $k2) (local.get $stride)))
            (local.set $a0 (v128.const i32x4 0 0 0 0))
            (local.set $a1 (v128.const i32x4 0 0 0 0))
            (local.set $a2 (v128.const i32x4 0 0 0 0))
            (local.set $a3 (v128.const i32x4 0 0 0 0))
            (local.set $kk (i32.const 0))
            (block $k_done
              (loop $k_loop
                (br_if $k_done (i32.ge_u (local.get $kk) (local.get $stride)))
                (local.set $pv (v128.load (i32.add (local.get $prow) (local.get $kk))))
                (local.set $a0 (f32x4.add (local.get $a0)
                  (f32x4.mul (local.get $pv) (v128.load (i32.add (local.get $k0) (local.get $kk))))))
                (local.set $a1 (f32x4.add (local.get $a1)
                  (f32x4.mul (local.get $pv) (v128.load (i32.add (local.get $k1) (local.get $kk))))))
                (local.set $a2 (f32x4.add (local.get $a2)
                  (f32x4.mul (local.get $pv) (v128.load (i32.add (local.get $k2) (local.get $kk))))))
                (local.set $a3 (f32x4.add (local.get $a3)
                  (f32x4.mul (local.get $pv) (v128.load (i32.add (local.get $k3) (local.get $kk))))))
                (local.set $kk (i32.add (local.get $kk) (i32.const 16)))
                (br $k_loop)))
            (f32.store offset=0 (local.get $optr)
              (f32.add
                (f32.add (f32x4.extract_lane 0 (local.get $a0)) (f32x4.extract_lane 1 (local.get $a0)))
                (f32.add (f32x4.extract_lane 2 (local.get $a0)) (f32x4.extract_lane 3 (local.get $a0)))))
            (f32.store offset=4 (local.get $optr)
              (f32.add
                (f32.add (f32x4.extract_lane 0 (local.get $a1)) (f32x4.extract_lane 1 (local.get $a1)))
                (f32.add (f32x4.extract_lane 2 (local.get $a1)) (f32x4.extract_lane 3 (local.get $a1)))))
            (f32.store offset=8 (local.get $optr)
              (f32.add
                (f32.add (f32x4.extract_lane 0 (local.get $a2)) (f32x4.extract_lane 1 (local.get $a2)))
                (f32.add (f32x4.extract_lane 2 (local.get $a2)) (f32x4.extract_lane 3 (local.get $a2)))))
            (f32.store offset=12 (local.get $optr)
              (f32.add
                (f32.add (f32x4.extract_lane 0 (local.get $a3)) (f32x4.extract_lane 1 (local.get $a3)))
                (f32.add (f32x4.extract_lane 2 (local.get $a3)) (f32x4.extract_lane 3 (local.get $a3)))))
            (local.set $optr (i32.add (local.get $optr) (i32.const 16)))
            (local.set $o (i32.add (local.get $o) (i32.const 4)))
            (br $out_loop)))
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $rows_loop)))))
