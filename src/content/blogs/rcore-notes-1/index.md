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

## 计算机组成基础

计算机主要由 CPU、物理内存和 I/O 外设组成。操作系统负责管理这些硬件资源，并为应用程序提供统一、易用的抽象。CPU 的主要任务是从物理内存中读取指令、译码并执行，同时与内存和 I/O 设备交换数据。

物理内存可以看成一个大的字节数组，每个字节都有对应的物理地址，CPU 通过地址访问其中的数据。多字节数据还涉及 **端序（Endianness）** 和 **地址对齐**：Little-endian 将低位字节放在低地址，x86、RISC-V 等常见架构通常采用这种方式；地址对齐则要求多字节数据的首地址满足一定倍数关系，例如 32-bit 数据通常按 4 字节对齐，以提高访问效率并避免某些架构上的异常。

![端序示意图](/assets/img/posts/rcore-notes-1/endianness-comparison.jpg)

## 了解 QEMU 模拟器

我们编写的内核主要在 QEMU 模拟器上运行来验证正确性。本笔记使用 `qemu-system-riscv64` 模拟一台 **64 位 RISC-V 计算机**。QEMU 会模拟 CPU、物理内存以及 I/O 外设，这些虚拟硬件最终仍然依赖宿主机（Linux / Windows / macOS）的真实硬件资源。QEMU 可以通过启动参数配置模拟计算机的硬件环境，例如 CPU、内存大小、外设以及启动方式。在实验中，QEMU 的性能已经足够用于运行和调试操作系统内核。启动 QEMU 并运行内核的命令如下：

```bash
qemu-system-riscv64 \
    -machine virt \
    -nographic \
    -bios ../bootloader/rustsbi-qemu.bin \
    -device loader,file=target/riscv64gc-unknown-none-elf/release/os.bin,addr=0x80200000
```

- `-machine virt` 表示使用 QEMU 提供的通用 RISC-V `virt` 虚拟机平台；
- `-nographic` 表示不启动图形界面，而是直接通过终端进行输入输出；
- `-bios ../bootloader/rustsbi-qemu.bin` 指定 QEMU 启动时首先运行的 Bootloader。这里使用预编译的 RustSBI，它负责完成基本初始化，并进一步启动操作系统内核；
- `-device loader` 用于在 QEMU 启动前，将宿主机上的内核镜像加载到模拟的物理内存中：`file=target/riscv64gc-unknown-none-elf/release/os.bin` 表示需要加载的内核镜像文件，而 `addr=0x80200000` 表示将 `os.bin` 加载到物理内存地址 `0x80200000`。

因此，整个启动过程可以概括为：`QEMU 启动 → RustSBI Bootloader → 加载 os.bin 到 0x80200000 → 跳转执行操作系统内核`。这里的 `os.bin` 是由内核可执行文件 `os` 进一步处理得到的裸二进制内核镜像，后续会继续介绍它的生成过程。

### QEMU 启动流程

在 QEMU 模拟的 `virt` 硬件平台上，物理内存的起始地址为 `0x80000000`，默认大小为 128 MiB，因此对应的物理地址范围为 `[0x80000000, 0x88000000)`。启动 QEMU 时，会预先把 Bootloader `rustsbi-qemu.bin` 加载到物理地址 `0x80000000`，同时把内核镜像 `os.bin` 加载到物理地址 `0x80200000`。

整个启动过程可以分成三个阶段：

1. **QEMU 固化的启动代码**：QEMU CPU 的程序计数器 PC 初始值为 `0x1000`，因此第一条指令从 `0x1000` 开始执行。完成必要的初始化后，这段代码会跳转到固定的物理地址 `0x80000000`。
2. **RustSBI Bootloader**：由于上一阶段会跳转到 `0x80000000`，因此 `rustsbi-qemu.bin` 必须被加载到这个位置。RustSBI 会完成进一步的硬件初始化，并在结束后将控制权交给操作系统内核。本实验使用的 RustSBI 约定下一阶段的内核入口地址为 `0x80200000`，因此 RustSBI 初始化完成后会跳转到该地址。
3. **操作系统内核**：为了能够与 RustSBI 正确衔接，内核镜像 `os.bin` 必须提前加载到 `0x80200000`，并保证内核的第一条指令位于该地址。当 CPU 开始执行 `0x80200000` 处的指令时，说明计算机控制权已经正式交给操作系统内核。

为了让内核镜像能够正确对接 QEMU 和 RustSBI，提交给 QEMU 的内核镜像文件必须满足：文件开头就是内核待执行的第一条指令。但通过移除标准库依赖得到的可执行文件实际上并不满足该条件。因此，还需要对可执行文件进行处理，才能得到可提交给 QEMU 的内核镜像。

## 程序内存布局与编译流程

### 程序内存布局

源代码被编译成可执行文件后，文件中的内容通常可以分为代码和数据两部分。代码部分由 CPU 可以解码并执行的指令组成，数据部分则是在程序运行过程中被 CPU 读写的内存内容。为了方便管理，编译器和链接器会进一步把这些内容划分成不同的段（Section），并将它们放置到不同的内存区域中，从而形成程序的内存布局（Memory Layout）。典型的程序内存布局从低地址到高地址大致如下：

<img
  src='/assets/img/posts/rcore-notes-1/program-memory-layout.jpg'
  alt='程序内存布局示意图'
  style='display: block; width: min(100%, 680px); height: auto; margin-inline: auto;'
/>

其中，代码部分主要位于 `.text` 段，用来保存程序编译后生成的机器指令。

数据部分通常还会继续划分：

- `.rodata`：保存只读的已初始化全局数据，例如常量、字符串常量等；
- `.data`：保存已经初始化、并且可以被修改的全局数据；
- `.bss`：保存未初始化的全局数据，程序加载时通常会将这部分区域初始化为 0；
- `heap`：用于保存程序运行期间动态分配的数据，例如 C/C++ 中通过 `malloc`、`new` 分配的对象。Heap 通常向高地址方向增长；
- `stack`：主要用于保存函数调用上下文、返回地址以及局部变量等。每次函数调用都会形成相应的栈帧，Stack 通常向低地址方向增长。

不同段具有不同的用途和访问权限，这种划分可以让程序的加载、执行和内存管理更加清晰。

### 编译流程

从源代码到可执行文件通常会经历编译、汇编和链接三个阶段：

1. **编译器（Compiler）**：将高级语言源文件转换为汇编代码。此时生成的仍然是文本形式的汇编文件。
2. **汇编器（Assembler）**：将汇编代码转换为机器码，生成二进制的 **目标文件（Object File）**。
3. **链接器（Linker）**：将多个目标文件以及可能使用到的外部目标文件合并，生成最终的可执行文件。

每个目标文件都有自己独立的程序内存布局，其中包含 `.text`、`.rodata`、`.data`、`.bss` 等 Section。链接器需要把多个目标文件中的这些 Section 整合成一个统一的内存布局。例如，两个目标文件 `1.o` 和 `2.o` 内部可能都拥有各自的 Section，而且这些 Section 在各自目标文件中的地址可能互相冲突。链接时，Linker 会根据 Section 的功能重新排列它们，将相同类型的 Section 合并到一起，并为它们重新分配地址，从而形成最终文件的统一内存布局：

![链接器合并 Section 的示意图](/assets/img/posts/rcore-notes-1/section-layout-rearrangement.jpg)

在模块化编程中，不同模块之间会通过全局变量、函数等进行相互访问。源代码中只需要使用变量名或函数名，这些名字称为符号（Symbol）。根据符号来自当前模块还是其他模块，可以分为内部符号和外部符号。在机器码中，CPU 不能直接通过符号名访问变量或函数，而必须使用具体地址。例如调用函数时，指令中最终需要保存函数入口的绝对地址，或者相对于当前 PC 的相对地址。因此，在生成最终可执行文件的过程中，需要把符号替换成实际地址。

当一个模块经过汇编生成目标文件后，模块内部各个 Section 的相对位置已经确定，因此内部符号通常已经能够确定对应的位置。但对于来自其他模块的外部符号，此时还无法知道其最终地址，因此目标文件会通过符号表（Symbol Table）记录这些符号的信息，供后续链接阶段处理。由于后续 Section 的位置还可能发生变化，内部符号的信息通常也会被记录下来。

外部符号的最终地址要到链接（Linking）阶段才能确定。链接器把多个目标文件合并后，各个 Section 在最终内存布局中的位置才会确定，此时不同模块之间引用的外部符号就可以被解析为具体地址。

链接过程中，各目标文件中的 Section 还会被重新排列和合并，因此即使是原本已经确定位置的内部符号，其最终地址也可能发生变化。链接器需要根据最终内存布局修正机器码中相关的地址，这个过程称为重定位（Relocation）。例如，模块 1 调用了模块 2 中定义的函数，在模块 1 单独生成目标文件时，只知道函数的符号名，并不知道函数最终会位于哪个地址。链接器将两个目标文件合并后，确定模块 2 中该函数的最终位置，再把模块 1 中对应的函数调用地址修正为正确值。

## 编写内核第一条指令

为了验证内核镜像是否能够正确接入 QEMU，首先编写进入内核后执行的第一条指令。在 `os/src/entry.asm` 中加入：

```asm
.section .text.entry
.globl _start

_start:
    li x1, 100
```

其中，`li x1, 100` 表示将立即数 `100` 加载到寄存器 `x1` 中。`li` 是 Load Immediate 的缩写。

`_start` 是一个符号，它指向紧跟在其后的指令，因此 `_start` 的地址就是 `li x1, 100` 这条指令所在的地址。`.globl _start` 将 `_start` 声明为全局符号，使其他目标文件在链接时也可以引用它。

`.section .text.entry` 表示将后续代码放入名为 `.text.entry` 的 Section。通常程序代码会放在 `.text` 段中，这里单独使用 `.text.entry`，是为了在后续链接时能够把它放在其他代码段之前，从而保证 `_start` 作为内核入口最先被执行。

为了让 Rust 编译器将这段汇编代码一起编译到内核中，需要在 `os/src/main.rs` 中嵌入 `entry.asm`：

```rust
#![no_std]
#![no_main]

mod lang_items;

use core::arch::global_asm;

global_asm!(include_str!("entry.asm"));
```

其中，`include_str!("entry.asm")` 会读取同目录下的 `entry.asm` 并将其转换为字符串，`global_asm!` 则将这段汇编代码作为全局汇编嵌入当前 Rust 程序中。

## 调整内核的内存布局

默认链接器生成的内存布局不能满足内核与 QEMU 对接的要求，因此需要通过链接脚本（Linker Script）自定义最终可执行文件中各个 Section 的位置。这里要求内核第一条指令位于物理地址 `0x80200000`。首先在 Cargo 配置中指定目标平台，并通过 `rustflags` 让链接器使用自定义的 `src/linker.ld`：

```toml
# os/.cargo/config
[build]
target = "riscv64gc-unknown-none-elf"

[target.riscv64gc-unknown-none-elf]
rustflags = [
    "-Clink-arg=-Tsrc/linker.ld",
    "-Cforce-frame-pointers=yes"
]
```

链接脚本 `os/src/linker.ld` 如下：

```ld
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

`OUTPUT_ARCH(riscv)` 指定最终可执行文件的目标架构为 RISC-V，`ENTRY(_start)` 将之前定义的全局符号 `_start` 设置为程序入口。

`BASE_ADDRESS = 0x80200000` 定义内核的起始地址，并通过 `. = BASE_ADDRESS;` 将链接器的当前位置 `.` 设置为 `0x80200000`。链接器之后会从该位置开始依次放置各个 Section。链接脚本中可以通过给 `.` 赋值来调整后续 Section 的位置，也可以将当前地址记录到全局符号中，例如 `stext = .;` 表示使用 `stext` 记录此时的地址。

Section 的基本写法为：

```ld
.rodata : {
    *(.rodata)
}
```

冒号前面的 `.rodata` 是最终可执行文件中的 Section 名称，花括号内部描述哪些输入目标文件中的 Section 会被放入其中。通配符 `*` 表示所有输入目标文件，因此 `*(.text .text.*)` 表示将所有目标文件中的 `.text` 以及 `.text.*` Section 合并到最终的 `.text` 中。

最终的内存布局按照 `.text`、`.rodata`、`.data`、`.bss` 的顺序从低地址向高地址排列，并通过 `stext`、`etext`、`srodata`、`erodata`、`sdata`、`edata`、`sbss`、`ebss` 等符号记录各个区域的起始或结束地址。

不同 Section 之间使用 `. = ALIGN(4K);` 将当前位置按照 4 KiB 对齐。`.text` 中首先放置 `*(.text.entry)`，然后才放置其他 `.text` Section：`*(.text .text.*)`。

由于整个内核从 `BASE_ADDRESS` 即 `0x80200000` 开始布局，因此 `.text.entry` 会位于整个代码段的最前面，从而保证 `_start` 对应的内核第一条指令位于 `0x80200000`，与 RustSBI 跳转到内核的地址正确对应。`.bss` 中首先放置 `.bss.stack`，之后再放置普通的 `.bss` 和 `.sbss` 数据。`/DISCARD/` 则用于丢弃不需要进入最终内核镜像的 Section，例如 `.eh_frame`。完成链接脚本配置后，可以重新编译内核：

```bash
cargo build --release
Finished release [optimized] target(s) in 0.10s
file target/riscv64gc-unknown-none-elf/release/os
target/riscv64gc-unknown-none-elf/release/os: ELF 64-bit LSB executable, UCB RISC-V, version 1 (SYSV), statically linked, not stripped
```

我们以 `release` 模式生成了内核可执行文件，它的位置在 `os/target/riscv64gc.../release/os`。接着通过 `file` 工具查看它的属性，生成的 `os` 是面向 64 位 RISC-V 架构的可执行文件，并采用静态链接。

## 手动加载内核可执行文件

前面生成的内核可执行文件虽然已经满足内存布局要求，但还不能直接交给 QEMU 加载。原因是可执行文件中除了真正需要运行的代码段和数据段之外，还包含一些额外的元数据。这些元数据主要用于描述可执行文件本身的结构和信息，QEMU 在直接加载内核镜像时并不需要它们。如果把整个可执行文件原样加载到内存中，这些额外内容会占据空间，并可能导致代码段和数据段被放到错误的物理地址。

<img
  src='/assets/img/posts/rcore-notes-1/kernel-image-metadata-comparison.jpg'
  alt='内核可执行文件与裸二进制镜像的元数据比较'
  style='display: block; width: min(100%, 850px); height: auto; margin-inline: auto;'
/>

图中红色区域表示内核可执行文件中的元数据，深蓝色区域表示各个段（包括代码段和数据段），浅蓝色区域则表示内核被执行的第一条指令，它位于深蓝色区域的开头。图示上半部分中，我们直接将内核可执行文件 `os` 提交给 QEMU；QEMU 会将整个可执行文件不加处理地加载到内存的 `0x80200000` 处。由于可执行文件开头是一段元数据，QEMU 内存的 `0x80200000` 处无法找到内核第一条指令，RustSBI 也就无法正常将计算机控制权转交给内核。相反，图示下半部分中，将元数据丢弃得到的内核镜像 `os.bin` 被加载到 QEMU 后，可以在 `0x80200000` 处正确找到内核第一条指令。

前面得到的内核可执行文件中除了实际需要加载的代码和数据之外，还包含符号表、重定位信息等元数据。由于 QEMU 的加载功能比较简单，需要先去除这些元数据，将内核可执行文件转换为裸二进制形式的内核镜像。使用以下命令生成内核镜像 `os.bin`：

```bash
rust-objcopy --strip-all target/riscv64gc-unknown-none-elf/release/os -O binary target/riscv64gc-unknown-none-elf/release/os.bin
```

可以使用 `stat` 工具比较内核可执行文件和内核镜像的大小：

```bash
$ stat target/riscv64gc-unknown-none-elf/release/os
File: target/riscv64gc-unknown-none-elf/release/os
Size: 1016             Blocks: 8          IO Block: 4096   regular file
...

$ stat target/riscv64gc-unknown-none-elf/release/os.bin
File: target/riscv64gc-unknown-none-elf/release/os.bin
Size: 4                Blocks: 8          IO Block: 4096   regular file
...
```

可以看到，生成的内核镜像 `os.bin` 只有 4 字节，这是因为当前内核中只有 `entry.asm` 中编写的一条指令，而一般情况下 RISC-V 架构的一条指令长度就是 4 字节。相比之下，内核可执行文件 `os` 大小为 1016 字节，因为其中还包含大量用于描述和加载可执行文件的元数据。这些元数据可以帮助系统在加载可执行文件时完成重定位、动态链接等工作，但 QEMU 无法直接利用这些信息，因此这里将它们去除，只保留真正需要加载到内存中的代码和数据。从某种意义上说，这相当于我们手动完成了一部分可执行文件的加载工作。

## 函数调用与栈

从汇编指令的角度看，如果 CPU 按顺序执行定长指令，设当前指令地址为 $a_n$，每条指令长度为 $L$ 字节，那么下一条指令地址通常满足：$a_{n+1}=a_n+L$。但程序并不总是顺序执行。遇到跳转指令时，CPU 会修改 `pc` 寄存器，使其跳转到指定地址，从而实现程序中的控制流（Control Flow），例如 `if/switch` 分支以及 `for/while` 循环。

函数调用（Function Call）是一种更特殊的控制流。调用函数时，需要先跳转到被调用函数的入口执行；函数执行结束后，还需要返回到调用指令的下一条指令继续执行。与普通分支不同，函数返回地址并不是编译期固定的。同一个函数可能在程序中的多个位置被调用，因此每次调用对应的返回地址都可能不同。这个返回地址只能在函数调用实际发生时确定。因此，实现函数调用不仅需要完成到被调用函数的跳转，还需要在调用时保存当前的返回地址，以便函数执行结束后能够恢复并跳回正确的位置。

<img
  src='/assets/img/posts/rcore-notes-1/library-function-call-stack-flow.jpg'
  alt='函数调用、返回地址与栈帧流转示意图'
  style='display: block; width: min(100%, 620px); height: auto; margin-inline: auto;'
/>

普通跳转只需要修改 `pc`，而函数调用还需要额外保存**返回地址**，这样函数执行结束后才能回到调用位置继续执行。RISC-V 中有两条常用于函数调用的跳转指令：

| 指令 | 指令功能 |
| --- | --- |
| `jal rd, imm[20:1]` | `rd ← pc + 4`<br>`pc ← pc + imm` |
| `jalr rd, (imm[11:0])rs` | `rd ← pc + 4`<br>`pc ← rs + imm` |

<div class="note-box">
  <p><strong>RISC-V 指令各部分含义</strong></p>
  <p>在大多数只与通用寄存器打交道的指令中，<code>rs</code> 表示 <strong>源寄存器（Source Register）</strong>，<code>imm</code> 表示 <strong>立即数（Immediate）</strong>，是一个常数，二者通常构成指令的输入部分；<code>rd</code> 表示 <strong>目标寄存器（Destination Register）</strong>，是指令的输出部分。</p>
  <p><code>rs</code> 和 <code>rd</code> 可以从 32 个通用寄存器 <code>x0~x31</code> 中选取，但这些部分并不是所有指令都必须具备；有些指令只有输入部分，也有些指令没有输出部分。</p>
</div>

## 函数返回与函数调用上下文

RISC-V 中用于函数调用的跳转指令除了修改 `pc` 完成跳转之外，还会把当前跳转指令的下一条指令地址保存到 `rd` 中：`rd ← pc + 4`。通常使用 `ra` 寄存器，也就是 `x1`，作为 `rd` 来保存这个返回地址。这样，当被调用函数执行结束后，只需要跳转回 `ra` 中保存的地址，就可以继续执行调用函数中的下一条指令。函数返回时通常使用伪指令 `ret`；汇编器会将它翻译为 `jalr x0, 0(x1)`。其中 `x1` 就是 `ra`，因此该指令会跳转到 `ra` 中保存的返回地址；而 `x0` 是恒为 `0` 的寄存器，所以 `jalr` 原本用于保存下一条指令地址到 `rd` 的操作会被直接丢弃。

因此，在函数调用时会保存返回地址并跳转到被调用函数，而函数返回时则通过 `ret` 跳转回原来的返回地址。这种方式在没有嵌套函数调用时可以正常工作，因为 `ra` 在函数执行期间不会被修改。但实际程序经常存在多层嵌套调用。例如函数 `f` 调用函数 `g` 时，调用 `g` 的跳转指令会再次把新的返回地址写入 `ra`，从而覆盖原来函数 `f` 的返回地址。这样当 `g` 返回后，`f` 原本应该返回的位置就已经丢失。

因此，要正确支持嵌套函数调用，就必须保证一个函数在调用其他函数前后，某些重要寄存器的值能够保持不变。这个要求不仅适用于 `ra`，还适用于其他通用寄存器。因为编译器会独立编译每个函数，一个函数无法预先知道它调用的其他函数会修改哪些寄存器，但部分寄存器的值可能又会影响当前函数后续的执行。由于函数调用发生前后需要保持的一组寄存器状态，被称为函数调用上下文（Function Call Context）。

由于 CPU 只有一套寄存器，而函数调用前后又需要保持函数调用上下文不变，因此必须借助物理内存保存寄存器状态。通常在调用子函数前，把需要保留的寄存器保存到内存中；函数执行结束后，再从相应位置读取并恢复这些寄存器。函数调用上下文中的寄存器通常分为两类：

- **被调用者保存（Callee-Saved）寄存器**：被调用函数可能会使用这些寄存器，因此如果要修改，必须由被调用函数自己先保存，并在返回前恢复，保证调用前后的值保持不变。
- **调用者保存（Caller-Saved）寄存器**：被调用函数可以自由覆盖这些寄存器。如果调用者希望某些值在函数调用后仍然保留，就需要在调用之前自行保存，并在函数返回后恢复。

因此，调用函数时，调用者会先保存自己不希望在调用过程中发生变化的 Caller-Saved 寄存器，然后通过 `jal` / `jalr` 调用子函数，并在返回后恢复这些寄存器。被调用函数进入后，则会保存自己执行过程中需要使用的 Callee-Saved 寄存器，完成函数执行后，在退出前恢复这些寄存器。这种在函数入口和退出位置进行寄存器保存与恢复的汇编代码，通常分别称为**开场（Prologue）**和**结尾（Epilogue）**，一般由编译器自动生成。一个函数既可能作为 Caller 调用其他函数，也可能作为 Callee 被其他函数调用，因此同一个函数中可能同时存在 Caller-Saved 和 Callee-Saved 寄存器的保存与恢复操作。
