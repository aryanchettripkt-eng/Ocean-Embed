import { gsap } from "gsap";

export function attachLandingMotion(root: HTMLElement) {
  const disposers: (() => void)[] = [];
  const media = gsap.matchMedia();
  media.add("(prefers-reduced-motion: no-preference)", () => {
    const context = gsap.context(() => {
      gsap.from("[data-entrance]", {
        y: 22,
        opacity: 0.35,
        duration: 1.15,
        stagger: 0.16,
        ease: "power3.out",
        clearProps: "all",
      });
    }, root);
    root.querySelectorAll<HTMLElement>("[data-micro]").forEach((element) => {
      const cta = element.dataset.micro === "cta";
      const animate = (scale: number, y: number) =>
        gsap.to(element, {
          scale,
          y,
          duration: 0.3,
          overwrite: "auto",
          ease: "power2.out",
        });
      const enter = () => {
        animate(cta ? 1.04 : 1.025, cta ? -3 : -1);
        gsap.to(element, {
          "--hover-light": 0.12,
          duration: 0.3,
          overwrite: "auto",
        });
        if (cta) {
          gsap.fromTo(
            element,
            { "--glint-x": "-65%" },
            {
              "--glint-x": "65%",
              duration: 0.85,
              ease: "power2.out",
              overwrite: "auto",
            },
          );
        }
      };
      const leave = () => {
        animate(1, 0);
        gsap.to(element, {
          "--hover-light": 0,
          duration: 0.35,
          overwrite: "auto",
        });
      };
      const down = () => animate(0.97, 0);
      const events: [string, EventListener][] = [
        ["pointerenter", enter],
        ["pointerleave", leave],
        ["focus", enter],
        ["blur", leave],
        ["pointerdown", down],
        ["pointerup", enter],
      ];
      events.forEach(([name, handler]) =>
        element.addEventListener(name, handler),
      );
      disposers.push(() => {
        events.forEach(([name, handler]) =>
          element.removeEventListener(name, handler),
        );
        gsap.killTweensOf(element);
        gsap.set(element, { clearProps: "transform,--hover-light,--glint-x" });
      });
    });
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (!entry.isIntersecting) return;
          context.add(() =>
            gsap.from(entry.target, {
              y: 20,
              opacity: 0.45,
              duration: 0.8,
              clearProps: "all",
            }),
          );
          observer.unobserve(entry.target);
        });
      },
      { threshold: 0.15 },
    );
    root
      .querySelectorAll("[data-reveal]")
      .forEach((element) => observer.observe(element));
    return () => {
      observer.disconnect();
      context.revert();
      disposers.splice(0).forEach((dispose) => dispose());
    };
  });
  return () => media.revert();
}
