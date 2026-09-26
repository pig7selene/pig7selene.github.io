---
title: 'rCore Notes 1 基本执行环境'
publishDate: 2026-09-23
category: learning
tags: [rCore, Operating Systems]
language: zh
description: 'Explains memory layout, linking, and the QEMU boot process, then builds and debugs a RISC-V kernel entry point with assembly and GDB.'
heroImage:
  src: /assets/img/posts/glove-model/glove-card-cover-v2.png
  alt: "White-haired figure with blue flowers against a pale-blue sky"
  color: "#78b8e5"
---

rCore 的训练营开始了。想到操作系统学得不是太好，所以来玩一下。

训练营地址：[openCamp rCore 训练营](https://opencamp.cn/os2edu/camp/2026fall)

rCore Book：[rCore-Tutorial-Book-v3](https://rcore-os.cn/rCore-Tutorial-Book-v3/index.html)

我们这一章的目标是构建一个内核最小执行环境，使得它能在 RV64GC 裸机上运行。我们先尝试一下把 `Hello, world!` 的目标平台换成 `riscv64gc-unknown-none-elf`。

<img src='/assets/img/posts/rcore-notes-1/no-std-error.png' alt='在 RISC-V 裸机目标上找不到 Rust 标准库的编译错误' style='display: block; width: min(100%, 900px); height: auto; margin: 1.5rem auto;' />

我们会发现提示 `can't find crate for std`，说明我们在目标平台上找不到 Rust 标准库 `std`，因此我们尝试把 `std` 库换成被裁剪过的 `core` 库。我们在 `main.rs` 的开头加上 `#![no_std]`，告诉 Rust 编译器不使用 `std`。

<img src='/assets/img/posts/rcore-notes-1/no-std-panic.png' alt='移除标准库后的 Panic 处理错误' style='display: block; width: min(100%, 900px); height: auto; margin: 1.5rem auto;' />

发现编译器又报错了。移除 `std` 后，`println!` 这类依赖标准库的功能就不能直接使用了。与此同时，默认的 Panic 处理机制也被移除。我们先来处理一下 Panic 的问题：`core` 中并没有提供 `panic!` 宏的实现，因此需要自己实现一个 `panic_handler`。

首先在 `os/src` 下创建 `lang_items.rs`，并加入一个最简单的 Panic 处理函数：

```rust
use core::panic::PanicInfo;

#[panic_handler]
fn panic(_info: &PanicInfo) -> ! {
    loop {}
}
```

其中 `#[panic_handler]` 表示当程序发生 Panic 时调用该函数，返回类型 `!` 表示该函数永远不会正常返回；这里暂时通过无限循环处理。随后在 `main.rs` 中引入这个模块：

```rust
#![no_std]

mod lang_items;
```

再重新编译一下试试。

<img src='/assets/img/posts/rcore-notes-1/no-main-warning.png' alt='编译成功后缺少入口符号 _start 的警告' style='display: block; width: min(100%, 900px); height: auto; margin: 1.5rem auto;' />

编译成功了，但是仍然有 warning：`cannot find entry symbol _start`。这说明虽然我们已经通过 `#![no_main]` 移除了 Rust 默认的 `main` 入口，但还没有为裸机程序提供真正的入口 `_start`。普通应用程序的启动过程通常由操作系统和语言运行时负责，而现在这些运行环境都不存在，因此接下来需要由我们自己编写内核的第一条指令，并将 `_start` 作为内核入口。

此外，最后的 `cannot execute binary file` 是因为我们生成的是 `riscv64gc-unknown-none-elf` 的 RISC-V 裸机程序，而当前 macOS 是 ARM64 平台，不能直接通过 `cargo run` 执行。之后需要借助 QEMU 模拟 RISC-V 平台来运行内核。

接下来在 `os/src` 下创建 `entry.asm`，定义内核入口：

```asm
.section .text.entry
.globl _start

_start:
    li x1, 100
```

- `.section .text.entry`：把下面的代码放到 `.text.entry` 这个代码段里。后面可以通过链接脚本把这个段放到整个内核代码的最前面。
- `.globl _start`：把 `_start` 声明为全局符号，这样链接器才能把它当成程序入口。
- `_start:`：定义一个名为 `_start` 的标签，也就是内核开始执行的位置。
- `li x1, 100`：这里的 `li` 是 load immediate，表示“加载立即数”。把立即数 `100` 写入寄存器 `x1`。之后可以通过 GDB 检查 `x1` 是否等于 `100`，用来验证内核是否真的从 `_start` 开始执行。

接着我们在 `main.rs` 中嵌入这段汇编代码，然后用链接脚本把 `_start` 放到 `0x80200000`。

```rust
// os/src/main.rs
#![no_std]
#![no_main]

mod lang_items;

use core::arch::global_asm;

global_asm!(include_str!("entry.asm"));
```

```toml
[build]
target = "riscv64gc-unknown-none-elf"

[target.riscv64gc-unknown-none-elf]
rustflags = [
    "-Clink-arg=-Tsrc/linker.ld",
    "-Cforce-frame-pointers=yes"
]
```

接下来创建 `src/linker.ld`，手动指定内核的内存布局。由于 QEMU 会将内核加载到 `0x80200000`，因此需要保证 `_start` 最终也位于这个地址。

```text
OUTPUT_ARCH(riscv)
ENTRY(_start)

BASE_ADDRESS = 0x80200000;

SECTIONS
{
    . = BASE_ADDRESS;
    skernel = .;

    stext = .;
    .text : {
        *(.text.entry)
        *(.text .text.*)
    }

    . = ALIGN(4K);
    etext = .;

    srodata = .;
    .rodata : {
        *(.rodata .rodata.*)
        *(.srodata .srodata.*)
    }

    . = ALIGN(4K);
    erodata = .;

    sdata = .;
    .data : {
        *(.data .data.*)
        *(.sdata .sdata.*)
    }

    . = ALIGN(4K);
    edata = .;

    .bss : {
        *(.bss.stack)
        sbss = .;
        *(.bss .bss.*)
        *(.sbss .sbss.*)
    }

    . = ALIGN(4K);
    ebss = .;
    ekernel = .;

    /DISCARD/ : {
        *(.eh_frame)
    }
}
```

完成 `entry.asm` 和 `linker.ld` 后，先使用 Release 模式编译内核：

```bash
cargo build --release
```

随后可以通过 `rust-readobj` 检查生成 ELF 的入口地址是否位于 `0x80200000`：

```bash
rust-readobj -h target/riscv64gc-unknown-none-elf/release/os
```

最后使用 `rust-objcopy` 去除 ELF Header、符号表等元数据，生成可以直接加载到 QEMU 中的裸二进制内核镜像：

```bash
rust-objcopy --strip-all \
  target/riscv64gc-unknown-none-elf/release/os \
  -O binary \
  target/riscv64gc-unknown-none-elf/release/os.bin
```

此时 `os` 是带有 ELF 元数据的可执行文件，而 `os.bin` 只保留真正需要加载到内存中的代码和数据。

接下来我们在 `os` 目录下启动 QEMU 并加载 RustSBI 和内核镜像：

```bash
qemu-system-riscv64 \
  -machine virt \
  -nographic \
  -bios ../bootloader/rustsbi-qemu.bin \
  -device loader,file=target/riscv64gc-unknown-none-elf/release/os.bin,addr=0x80200000 \
  -s \
  -S
```

随后在另一个终端连接 GDB：

```bash
riscv64-elf-gdb \
  -ex 'file target/riscv64gc-unknown-none-elf/release/os' \
  -ex 'set arch riscv:rv64' \
  -ex 'target remote localhost:1234'
```

GDB 连接 QEMU 后，可以看到当前 `PC = 0x1000`，说明 CPU 会先执行 QEMU 的启动固件。通过 `x/10i $pc` 可以查看这段 Boot ROM 的指令，随后它会把 RustSBI 的入口地址 `0x80000000` 加载到寄存器中，并通过 `jr t0` 跳转过去。

<img src='/assets/img/posts/rcore-notes-1/qemu-boot-rom.png' alt='GDB 中 QEMU Boot ROM 的启动指令' style='display: block; width: min(100%, 400px); height: auto; margin: 1.5rem auto;' />

我们通过单步调试来复盘一下。

<img src='/assets/img/posts/rcore-notes-1/qemu-single-step.png' alt='GDB 单步调试 QEMU 启动过程' style='display: block; width: min(100%, 360px); height: auto; margin: 1.5rem auto;' />

当执行完 `0x1010` 处的 `ld t0,24(t0)` 后，寄存器 `t0` 被加载为 `0x80000000`；随后再执行 `jr t0`，PC 跳转到 `0x80000000`，说明控制权已经从 QEMU Boot ROM 转交给 RustSBI。

<img src='/assets/img/posts/rcore-notes-1/kernel-entry-breakpoint.png' alt='GDB 在内核入口 _start 处命中断点' style='display: block; width: min(100%, 400px); height: auto; margin: 1.5rem auto;' />

在内核入口 `0x80200000` 处设置断点并继续执行后，程序成功停在 `_start`。反汇编可以看到第一条指令是 `li x1, 100`；单步执行后 `x1 = 100`，说明内核已经正确加载并执行。同时此时 `sp = 0`，说明内核栈尚未初始化。
