---
title: 'rCore | 批处理系统'
publishDate: 2026-10-20
category: learning
tags: [rCore, Operating Systems]
language: zh
description: 'Builds a RISC-V batch operating system with application loading, user/kernel privilege switching, system calls, and trap handling.'
heroImage:
  src: /assets/img/posts/glove-model/glove-card-cover-v2.png
  alt: "White-haired figure with blue flowers against a pale-blue sky"
  color: "#78b8e5"
---

这一章尝试实现一个能依次加载、执行应用程序的批处理系统。先准备好应用程序和用户库，后面再让自己的内核接管它们。

## 实现应用程序

我们先把 Hello World 的输出在 `user` 目录下执行 `make build`，生成 RISC-V ELF 文件和对应的 `.bin` 镜像。

应用仍然使用 `#![no_std]` 和 `#![no_main]`，启动、打印和退出由 `user_lib` 提供。先来看一下用户库的入口：

```rust
#[unsafe(no_mangle)]
#[unsafe(link_section = ".text.entry")]
pub extern "C" fn _start() -> ! {
    clear_bss();
    unsafe {
        exit(main());
    }
    panic!("unreachable after sys_exit!");
}
```

程序从 `_start` 开始，先清空需要零初始化的 `.bss`，再调用应用的 `main()`，最后把返回值交给 `exit`。正常退出后不会再返回这里，因此后面的 `panic!` 只是兜底。

`link_section` 把入口代码放入 `.text.entry`。在 `user/src/linker.ld` 中，将基地址设为 `0x80400000`，优先放置这个段，并通过 `ENTRY(_start)` 指定入口。我们检查一下编译结果：

<img src="/assets/img/posts/rcore-notes-2/entry-address.png" alt="image-20261020120803961" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

输出 `start address: 0x0000000080400000`，和预期一致

反汇编中还能找到调用 `<main>` 的 `jalr`，以及设置 `a7 = 0x5d` 后的 `ecall`。这里 `0x5d` 就是十进制 `93`，对应退出系统调用

<img src="/assets/img/posts/rcore-notes-2/main-and-exit.png" alt="反汇编中调用 main 并通过 ecall 退出" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

接着看打印是怎么完成的，`sys_write` 把文件描述符、缓冲区地址和字节数交给 `syscall`，后者用内联汇编准备寄存器：

```rust
asm!(
    "ecall",
    inlateout("x10") args[0] => ret,
    in("x11") args[1],
    in("x12") args[2],
    in("x17") id
);
```

其中 `x10～x12` 对应 `a0～a2`，保存三个参数；`x17` 对应 `a7`，保存系统调用编号。对于输出，编号是 `64`，`fd = 1` 表示标准输出。调用结束后，`a0` 又用于保存返回值，所以这里使用 `inlateout`。

接下来运行 Hello World，成功输出了修改后的内容，然后执行 `echo $?`，得到 `0`，说明程序正常退出

<img src="/assets/img/posts/rcore-notes-2/hello-world.png" alt="Hello World 的正常输出及退出状态" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

随后把其余应用也运行一下：

| 应用 | 操作 | 运行结果 |
| --- | --- | --- |
| `01store_fault` | 向地址 `0` 写入一个字节 | `Segmentation fault`，退出 `139` |
| `02power` | 反复计算并输出结果 | `Test power OK!`，退出 `0` |
| `03priv_inst` | 在用户态执行 `sret` | `Illegal instruction`，退出 `132` |
| `04priv_csr` | 在用户态修改 `sstatus` | `Illegal instruction`，退出 `132` |

<img src="/assets/img/posts/rcore-notes-2/protection-tests.png" alt="非法访存、执行特权指令和访问特权寄存器的运行结果" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

这里出错的三个程序是故意测试非法操作的，目前这些应用由 QEMU 用户态模拟和 Linux 环境支持，后面再由我们自己的内核负责加载、处理系统调用和异常。

## 实现批处理操作系统

接下来先解决内核怎么找到应用的问题。我们在 `os/src/link_app.S` 中，通过 `.incbin` 把应用的 `.bin` 镜像放进内核的 `.data` 段：

```asm
app_0_start:
    .incbin "../user/target/riscv64gc-unknown-none-elf/release/00hello_world.bin"
app_0_end:
```

随后在内核入口文件中引入这段汇编：

```rust
global_asm!(include_str!("link_app.S"));
```

镜像前面还放了一张地址表，由 `_num_app` 指示它的位置。第一个整数是应用数量，后面依次保存各个应用的起始地址，最后额外保存最后一个应用的结束地址。每项通过 `.quad` 存放，占 8 字节。因此 5 个应用需要 6 个地址，相邻地址相减就能得到对应应用的长度。

检查内核的数据段，开头的 `05000000 00000000` 就是数量 `5` 的小端表示。

<img src="/assets/img/posts/rcore-notes-2/batch-data-table.png" alt="内核数据段中的应用数量与地址表" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

接着在 `os/src/batch.rs` 中定义应用管理器：

```rust
pub struct AppManager {
    num_app: usize,
    current_app: usize,
    app_start: [usize; MAX_APP_NUM + 1],
}
```

初始化时读取 `_num_app` 得到数量，再跳过这个字段，把后面的地址复制到 `app_start`。`current_app` 从 `0` 开始，记录当前应用编号。

我们先把这些信息打印出来，确认内核已经能找到应用。

<img src="/assets/img/posts/rcore-notes-2/batch-app-info.png" alt="AppManager 解析并打印五个应用的地址范围" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

然后实现 `load_app`：清空从 `0x80400000` 开始的 128 KiB 应用区域，将指定镜像复制过去，并检查复制后的内容是否一致。核心操作是：

```rust
app_dst.copy_from_slice(app_src);
asm!("fence.i");
```

其中 `fence.i` 保证后续取指能看到刚写入的代码。这里需要区分镜像的存放地址和运行地址：镜像原本作为数据放在内核里，加载时才复制到约定的 `0x80400000`。

后面的系统调用和异常处理也需要访问管理器，因此我们把它改成全局实例：

```rust
lazy_static! {
    static ref APP_MANAGER: UPSafeCell<AppManager> = unsafe {
        UPSafeCell::new(AppManager::new())
    };
}
```

`lazy_static!` 在第一次使用时初始化管理器，`UPSafeCell` 则封装 `RefCell`，在当前单核、无并发访问同一个 Cell 的前提下，通过 `exclusive_access()` 获取独占借用。借用检查在运行时进行。内核启动时先清空 `.bss`，再初始化并使用这个全局管理器。

重新编译、生成内核镜像并启动 QEMU 后，可以看到：

<img src="/assets/img/posts/rcore-notes-2/batch-app-loaded.png" alt="内核成功将应用 0 的 4608 字节加载到 0x80400000" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

最后把手写地址表改成由 `os/build.rs` 自动生成。脚本扫描用户程序、按文件名排序，再生成数量、地址表和 `.incbin` 指令。它在主机上运行，因此可以使用 `std`，Cargo 会在构建内核时自动执行它。

先在 `user` 目录执行 `make build`，再构建内核，就能得到包含最新应用镜像的内核。

<img src="/assets/img/posts/rcore-notes-2/batch-generated-table.png" alt="构建脚本自动生成应用数量、地址表和镜像嵌入指令" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

此时已经完成应用的放置和加载，应用还没有开始执行，后面再接上特权级切换。

## 实现特权级的切换

上一节只是把应用复制到了 `0x80400000`，接下来让它在用户态运行，并由内核接管系统调用和异常。

我们先在 `os/src/trap/context.rs` 中定义 Trap 上下文：

```rust
#[repr(C)]
pub struct TrapContext {
    pub x: [usize; 32],
    pub sstatus: Sstatus,
    pub sepc: usize,
}
```

它为通用寄存器预留 32 个槽位，再加上 `sstatus` 和 `sepc`，共占 `34 × 8 = 272` 字节。`#[repr(C)]` 固定字段的布局，方便汇编按偏移保存和恢复。

随后准备两个 8 KiB 的栈，分别供内核和用户程序使用。初始上下文中，将 `sepc` 设为应用入口 `0x80400000`，`x[2]` 设为用户栈顶，`sstatus.SPP` 设为 `User`。此时只是准备好了快照，还没有真正切换特权级。

<img src="/assets/img/posts/rcore-notes-2/trap-context-init.png" alt="Trap 上下文大小为 272 字节，并准备好用户栈和初始入口" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

接着在 `os/src/trap/trap.S` 中实现保存与恢复。`__alltraps` 是 Trap 入口，先通过下面的指令交换用户栈指针和预先存放在 `sscratch` 中的内核栈指针：

```asm
csrrw sp, sscratch, sp
```

然后在内核栈上保存通用寄存器、`sstatus`、`sepc` 和原来的用户栈指针，将上下文地址传给 Rust 的 `trap_handler`。我们把 `__alltraps` 的地址写入 `stvec`，这样发生 Trap 时，CPU 就会跳到这里。

`__restore` 则反过来恢复现场，换回用户栈，最后执行 `sret`，按 `SPP` 指定的特权级跳转到 `sepc`。第一次启动应用也复用这条路径：把刚构造的上下文放到内核栈上，再调用 `__restore`，就能进入 U 模式。

我们先让处理函数只打印现场，捕获到了来自 `User` 的 `Exception(8)`，以及 `a7 = 64`、`a0 = 1`。这正是 Hello World 发出的标准输出请求，说明用户程序已经运行，并通过 `ecall` 回到了内核。

<img src="/assets/img/posts/rcore-notes-2/trap-first-ecall.png" alt="捕获用户态第一次 ecall，系统调用编号为 64" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

接下来根据 `scause` 分发 Trap。对于 `UserEnvCall`，取出系统调用编号和参数，处理后把返回值写回 `a0`：

```rust
cx.sepc += 4;
let result = crate::syscall::syscall(
    cx.x[17],
    [cx.x[10], cx.x[11], cx.x[12]],
);
cx.x[10] = result as usize;
```

这里必须让 `sepc` 增加 4，因为它原本指向触发异常的 `ecall`，而返回时应该从下一条指令继续。处理函数最后返回上下文地址，汇编便能继续执行 `__restore`。

在 `os/src/syscall.rs` 中，编号 `64` 对应 `sys_write`：读取应用传来的缓冲区，通过 SBI 逐字节输出，再返回写入的字节数。编号 `93` 对应 `sys_exit`：打印退出码，运行下一个应用。我们将 Hello World 的输出改为英文后，得到：

<img src="/assets/img/posts/rcore-notes-2/trap-hello-exit.png" alt="" style="display: block; width: auto; max-width: min(100%, 900px); height: auto; margin: 1.5rem auto;" />

最后接上 `run_next_app`。它按编号加载应用，更新下一个待加载的编号，释放管理器的借用，再构造上下文并启动。这里要先 `drop(manager)`，否则进入用户态后借用不会自动释放，下一次切换就无法再次借用管理器。

对于非法存储和非法指令，也调用 `run_next_app`，不再返回出错的应用。重新编译并启动后，五个应用依次执行：

| 应用 | 内核处理结果 |
| --- | --- |
| `00hello_world` | 输出 Hello，正常退出，返回值为 `0` |
| `01store_fault` | 报告 `Store fault`，终止该应用并继续 |
| `02power` | 输出 `Test power OK!`，正常退出，返回值为 `0` |
| `03priv_inst` | 报告 `Illegal instruction`，终止该应用并继续 |
| `04priv_csr` | 报告 `Illegal instruction`，终止该应用并继续 |

最后打印 `[kernel] All applications completed!`，通过 SBI 请求关机，QEMU 自动退出并回到终端提示符。

<img src="/assets/img/posts/rcore-notes-2/trap-batch-completed.png" alt="五个应用依次运行，内核处理异常后继续，最终完成批处理并自动关机" style="display: block; width: auto; max-width: min(100%, 480px); height: auto; margin: 1.5rem auto;" />
